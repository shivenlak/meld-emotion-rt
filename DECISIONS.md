# DECISIONS

Each major decision: options considered → choice → evidence → trade-off.
Numbers come from scripts in this repo (file named next to each one).

---

## 1. Primary track: audio (speech), with ASR text as a second view of the same signal

- **Options:** text-only (MELD gold transcripts), audio, vision.
- **Choice:** audio. A person talks to a tabletop robot, so speech is the input that actually exists at
  deployment. From one microphone signal we get two views: *what* was said (ASR → text) and *how* it
  was said (raw audio → tone).
- **Trade-off:** text is the strongest single MELD signal, but in deployment text only exists via ASR,
  so an "audio track" that includes ASR is the honest version of a text system. We measure that gap
  explicitly (gold vs ASR transcript rows in the results table).

## 2. Real-time definition: turn-based conversational latency

- **Options:** (a) frame-level continuous emotion every N ms; (b) turn-level: react after each utterance.
- **Choice:** (b). MELD labels are per utterance, and a conversational robot replies per turn. Continuous
  per-frame emotion would have no ground truth to evaluate against.
- **Definition:** after the VAD detects end of speech (it waits 500 ms of silence), emotion tags (JSON)
  should be out within ~300 ms + ASR time, and the first reply token within ~1 s. We report p50/p95
  measured on real MELD test clips (`scripts/benchmark_latency.py`).
- **Trade-off:** the 500 ms silence wait is a fixed cost. Shorter means the robot interrupts people
  mid-pause. It is reported separately as `vad_wait`.

## 3. ASR: Whisper small.en via faster-whisper, on the experimental MPS backend

- **Options:** base.en (74M) vs small.en (242M); CPU int8 vs CTranslate2's experimental MPS backend.
- **Evidence (3 s clip, median of 5 runs, M4 Pro):**

  | model | CPU int8 | MPS fp16 |
  |---|---|---|
  | base.en | 286 ms | 106 ms |
  | small.en | 950 ms | 264 ms |

  All four gave the identical transcript on that clip.
- **Choice:** small.en on MPS. MELD is noisy TV audio (laugh tracks, overlapping speech), so the larger
  model is worth 160 ms. CPU fallback is automatic if MPS fails.
- **Trade-off:** the MPS backend is labelled experimental by CTranslate2.

## 4. Encoders frozen + cached, small trained head

- **Options:** fine-tune RoBERTa/WavLM end to end vs. freeze them and train a small head on cached
  embeddings.
- **Choice:** freeze. Embeddings are computed once (~10 min for all of MELD), then each head trains in
  seconds-to-minutes on CPU, so 3 seeds × several configs is affordable inside a 2–3 day timebox. The
  live pipeline uses exactly the same frozen features, so there's no train/serve mismatch.
- **Pooling:** mean over time of *every* layer's hidden states ([13, 768] per input). The head learns a
  softmax weight per layer (the SUPERB-benchmark recipe). Speech emotion info sits mostly in middle
  layers, not the last one.
- **Trade-off:** fine-tuning RoBERTa on MELD typically adds a few points of weighted F1. Left out
  (see README "intentionally left out").

## 5. Audio encoder: WavLM-base-plus

- **Options:** wav2vec2-base, HuBERT-base, WavLM-base-plus (all ~95M), emotion2vec (needs FunASR), larger
  (300M+) models.
- **Choice:** WavLM-base-plus. It's the best base-size model on the SUPERB emotion task, it's in
  `transformers` (no extra framework), and it's MIT-licensed.
- **Trade-off:** emotion2vec is emotion-specific and likely stronger. Not tried, for dependency/time reasons.

## 6. LLM: Qwen3-1.7B, 4-bit, via MLX

- **Options:** Qwen2.5-1.5B/3B, Qwen3-1.7B/4B, Llama-3.2-1B/3B; runtimes llama.cpp / MLX / transformers.
- **Choice:** Qwen3-1.7B (Apache-2.0) in MLX 4-bit. MLX is Apple's native framework, so it's fast on
  M-series. First token measured at ~90–120 ms on a short prompt.
- **Rejected:** Qwen2.5-3B uses the *Qwen Research* licence (non-commercial), a bad fit for a startup.
- **Thinking mode** is disabled (`enable_thinking=False`). Otherwise Qwen3 emits a long `<think>` block
  before replying.

## 7. Phase-1 off-the-shelf text classifier (j-hartmann) is NOT reported on MELD

Its training data includes MELD, so its MELD-test numbers would be inflated by leakage. It exists only
as a fallback so the pipeline runs before the head is trained.

## 8. Parameter counting

`scripts/count_params.py` loads the PyTorch models and sums `p.numel()`. Whisper (CTranslate2) and Qwen
(4-bit MLX) are counted from the original checkpoints' safetensors headers. For Qwen3-1.7B the header
says 2.03B, but the input embedding and `lm_head` are *tied* (one shared matrix, `tie_word_embeddings:
true`) and the checkpoint stores it twice. Unpacking the MLX 4-bit weights gives 1.72B unique weights.
We count the conservative 2.03B against the cap.

## 9. Class imbalance and modality dropout (dev-only sweep)

- **Options:** class-weight power p ∈ {0, 0.5, 1} (loss weight per class = (1/frequency)^p; 0 = none,
  1 = full inverse frequency) × modality dropout ∈ {0, 0.2, 0.4} (randomly zeroing one modality's
  features during training so the head can't ignore audio).
- **Evidence:** `scripts/sweep_fused.sh` → `results/sweep_fused_dev.md`. Dev weighted F1 (3 seeds):
  p=1 is clearly worse (~0.588). Every p ∈ {0, 0.5} cell is within ~1 std of the others (0.606–0.616).
- **Choice:** keep the defaults chosen *before* the sweep (p = 0.5, dropout 0.2). The best cell (p=0,
  dropout 0: 0.616 ± 0.007) beats them by less than the seed-to-seed noise, and picking it would be fitting
  noise on a 1,109-utterance dev set. p = 0.5 also had the best dev macro F1 (rare emotions matter for a robot).
- **Trade-off:** weighted F1 slightly favours not re-weighting at all, because neutral dominates it.

## 10. ASR decoding: no temperature fallback, token cap

- **Problem found in the latency benchmark:** one clip took 5.4 s for ASR. Root cause, reproduced: the
  VAD-cut segment ended in exact digital zeros. Whisper's greedy decode failed faster-whisper's quality
  checks, so it re-decoded at temperatures 0.4/0.6/0.8, once looping "Okay. Okay. Okay…" (3–7 s).
- **Choice:** `temperature=0.0` (decode once), `without_timestamps=True`, and `max_new_tokens` ≈ 10 per
  second of audio (capped at 400). Same clip: ~270 ms, correct transcript. File-based tests now pad with a
  quiet noise floor instead of zeros, because a real mic never outputs exact zeros.
- **Evidence it didn't hurt accuracy:** test WER 35.9% → 35.0%; fused-ASR weighted F1 0.529 → 0.530.
  ASR on the whole test set went from 825 s to 699 s. Latency p95 for EOS → tags went from 448 ms to 380 ms
  (max from 5,482 ms to 449 ms).
- **Process note:** this change was made for latency, after the first test run. To keep the reported
  ASR rows faithful to the shipped system I regenerated the test transcripts and re-ran the test evaluation
  once. The earlier table is kept in `results/results_test_asr_with_fallback.md`.

## 11. Learned MLP fusion vs. late fusion

- **Options:** MLP over concatenated projections (with auxiliary per-modality heads) vs. weighted average of
  the text-only and audio-only models' probabilities (weight picked on dev).
- **Evidence:** dev 0.610 vs 0.612 (tie). Test: gold 0.620 vs 0.624 (within noise), ASR 0.530 vs 0.539
  (late fusion slightly better).
- **Choice:** MLP, decided on dev before the test run. It's one model and it directly provides the
  per-modality probabilities for the JSON and for the LLM's "trust the tone" hint.
- **Trade-off / honest note:** late fusion is simpler and at least as good. If I continued, I'd switch to it
  and re-validate on dev.

## 12. "Same words, different tone" measured, not just demoed

- `scripts/same_words_test.py` uses the 155 test utterances whose exact text occurs with ≥ 2 gold emotions.
  Text-only must answer identically for identical text (ceiling 0.665). Result: fused 0.609 ± 0.015 vs
  text 0.613 ± 0.005, so no aggregate gain from tone. Early bug: the first version stripped punctuation,
  which merged "Hey!" with "Hey." and gave text-only an accuracy *above* the theoretical ceiling. Grouping
  by exact text fixed it.

## 13. Emotion state smoothing: EMA with alpha 0.7

- First tried alpha 0.6 on the two-turn TTS sample: after "I got the job!" (surprise 0.94) then "my dog is
  sick" (fear 0.60), the smoothed state stayed "surprise". That's too sticky for a robot that should react
  to bad news. alpha 0.7 switches to fear on the second turn and still damps a single noisy turn. A heuristic,
  not tuned on data (MELD has no "robot mood" labels).

## 14. Vision extension: off-the-shelf face model, late fusion, weight chosen on dev

- **Options:** (a) train a face head on MELD frames; (b) off-the-shelf facial-expression model + late fusion.
- **Choice:** (b). `trpakov/vit-face-expression` (ViT-base, 85.8M, FER2013) predicts exactly MELD's 7
  labels, so no retraining is needed and there's no MELD leakage. Training on MELD frames would need
  speaker-face attribution (who is talking in each shot), which MELD doesn't provide.
- **Face detector:** OpenCV 5.1 no longer ships Haar cascades, so I used YuNet (53k params, MIT), which
  OpenCV runs natively via `cv2.FaceDetectorYN`. It found a face in 99.1% of MELD dev/test clips.
- **Fusion:** p = (1 − w)·p_text_audio + w·p_face. w picked on dev from 0…0.6 per seed → [0.1, 0.1, 0.2].
  Deployed w = 0.1 (seed 0's, matching the deployed head).
- **Evidence:** `scripts/evaluate_vision.py` → `results/results_vision.md`. Face-only test weighted F1 0.267
  (below the 0.313 always-neutral baseline). Fused with ASR text: 0.530 → 0.533. With gold text: 0.620 → 0.623.
- **Trade-off:** +86M params and a background thread for ~+0.3 F1 on MELD. It's kept as an opt-in
  flag (`--vision`), not on by default. It might be worth more with a webcam facing one person (closer to
  FER2013 framing), but that wasn't measured.
- **Live design:** the face is scored continuously in a background thread (~4 fps). At end of speech we
  average over the utterance window, so vision adds no wait to the turn.
