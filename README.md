# Lumi: real-time speech emotion → robot state + reply (MELD)

A local, turn-based prototype for an expressive desk-lamp robot. A person speaks. The system hears
**what** they said (Whisper ASR → RoBERTa) and **how** they said it (WavLM on raw audio), fuses both
into one of MELD's 7 emotions, and emits:

1. a **structured JSON state** per turn (emotion, confidence, per-modality probabilities, smoothed mood,
   a robot expression tag, per-stage latency), and
2. a **short in-character reply**, streamed token by token from a local 1.7B LLM, grounded in the
   transcript and the detected emotion.

Everything runs locally on a MacBook (Apple M4 Pro): **2.49B total parameters** (2.58B with the
optional vision extension; the cap is 6B), with no remote inference. An optional **vision extension** adds
the speaker's facial expression (webcam or video) as a third modality.

**At a glance** (every number is produced by a script in `scripts/`, see [Results](#results-meld-official-splits)):

| | |
|---|---|
| MELD test weighted F1, live setting (Whisper transcript + audio) | **0.530** (text-only on the same transcripts: 0.515; gold-transcript text: 0.620) |
| End of speech → emotion JSON / first reply token (p50, M4 Pro) | **288 ms / 378 ms** (+ 500 ms silence wait to detect end of speech) |
| Parameters on the inference path | **2.493B** core, **2.579B** with vision (cap 6B) |
| Honest headline | Tone helps a little when the transcript is noisy (+1.5 F1). The face model helps even less (+0.3). Speech-recognition errors (35% WER on MELD) cost the most (−10.5 F1) |

This was built as a 2–3 day take-home. [DECISIONS.md](DECISIONS.md) explains each design choice with its
evidence, and [PROGRESS.md](PROGRESS.md) is the working log.

```
mic / file ─► Silero VAD ─► Whisper small.en ─► transcript ─► RoBERTa-base (frozen) ─┐
  (32 ms chunks)  (end of     (MPS)                                                   ├─► fusion head ─► 7 emotions
                  utterance)       └──────────── raw audio ─► WavLM-base-plus (frozen)┘   (trained, 0.5M)     │
                                                                                                             ▼
                                         JSON state ◄── state tracker (EMA over turns) ◄────────────────────┤
                                         streamed reply ◄── Qwen3-1.7B 4-bit (MLX) ◄── persona + transcript + emotion
```

## Quick start

Tested on macOS 15.5, Apple M4 Pro, 24 GB. Needs Homebrew `python@3.11` and `ffmpeg`.
The code blocks below contain only commands (no `#` comments), so they can be pasted straight into
zsh, the macOS default shell. zsh doesn't treat `#` as a comment when you type or paste commands.

**Setup (once):**

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Every new terminal:** activate the environment. Your prompt then starts with `(.venv)`, and `python`
means the project's Python 3.11. Without this, `python3` is macOS's own 3.9, which has no numpy.

```bash
source .venv/bin/activate
```

**Demos and tests** (no MELD needed: the trained heads ship in `models/`, 14 MB):

| command | what it does |
|---|---|
| `python run_file.py samples/two_turns_tts.wav` | file → VAD → turns → JSON + streamed reply |
| `python run_live.py` | live microphone: speak, pause ~0.5 s, it replies (Ctrl+C quits; terminal needs mic permission) |
| `python run_live.py --simulate samples/two_turns_tts.wav` | replays a file through the exact streaming path, in real time |
| `python run_file.py test:dia199_utt13` | any MELD clip by `split:id` (needs the MELD data) |
| `python run_live.py --vision` | live mic + **webcam**: face expression as a third modality (terminal needs camera permission) |
| `python run_file.py test:dia45_utt4 --vision` | MELD clip with face expression from its video frames |
| `python run_live.py --simulate clip.mp4 --vision` | replays a video's audio *and* frames in real time through the live path |
| `pytest -q` | 5 tests, ~10–40 s (the first run is slower) |

Library log noise is hidden by default. Add `--verbose` to any demo to see it.

If `models/fused.pt` is missing, the pipeline falls back to an off-the-shelf text classifier
(DECISIONS.md section 7).

**Full reproduction (MELD → every number in this README)**, in order. Times are measured on the M4 Pro:

```bash
./scripts/download_meld.sh
python scripts/extract_audio.py
python scripts/cache_embeddings.py
python scripts/cache_embeddings.py --skip-gold --asr-test
python scripts/train_head.py --modalities text --name text
python scripts/train_head.py --modalities audio --name audio
python scripts/train_head.py --modalities text audio --name fused
./scripts/sweep_fused.sh
python scripts/evaluate.py --split dev
python scripts/evaluate.py --split test
python scripts/count_params.py
python scripts/benchmark_latency.py --n 100
python scripts/same_words_test.py
python scripts/qualitative_examples.py
python scripts/cache_vision.py
python scripts/evaluate_vision.py
```

| step | what / time |
|---|---|
| `download_meld.sh` | ~10.9 GB over 8 parallel connections (~20 min), then extracts. Afterwards you can delete `data/download/MELD.Raw.tar.gz` |
| `extract_audio.py` | mp4 → 16 kHz mono wav (~1 min, 1.3 GB) |
| `cache_embeddings.py` | frozen RoBERTa + WavLM embeddings for train/dev/test (~10 min) |
| `cache_embeddings.py --skip-gold --asr-test` | Whisper on the 2,610 test clips + their text embeddings (~12 min) |
| `train_head.py` ×3 | text / audio / fused heads, 3 seeds each (~25 s / ~20 s / ~40 s) |
| `sweep_fused.sh` | optional dev-only sweep (~5 min) |
| `evaluate.py --split test` | the one test run → `results/results_test.md` |
| the rest | params, latency, same-words test, qualitative examples → `results/` |
| `cache_vision.py` / `evaluate_vision.py` | vision extension: face probabilities for dev/test (~10 min), then the face weight chosen on dev and a test evaluation |

Models download automatically from Hugging Face on first use (~2.5 GB total).

## Example output (one turn)

`python run_file.py test:dia199_utt13` (Rachel, gold label *sadness*; text alone says *neutral*, tone says *sadness*):

```json
{"transcript": "I guess I just figured that somewhere down the road we would be on again.",
 "emotion": "sadness", "confidence": 0.639,
 "per_modality": {
   "text":  {"anger": 0.035, "disgust": 0.016, "fear": 0.018, "joy": 0.07,  "neutral": 0.646, "sadness": 0.207, "surprise": 0.008},
   "audio": {"anger": 0.017, "disgust": 0.027, "fear": 0.063, "joy": 0.093, "neutral": 0.127, "sadness": 0.631, "surprise": 0.042}},
 "smoothed_state": "sadness", "expression": "droop_dim_blue",
 "latency_ms": {"asr": 272.0, "emotion": 88.4, "llm_first_token": 100.4, "llm_total": 170.9}}
Lumi: Hmm, that feels sad. Maybe we'll be on again someday.
```
(Copied from `results/qualitative_examples.json`. The full state also has `turn_id`, fused `probs`,
`smoothed_confidence`, `audio_seconds` and `classifier`.)

## Results (MELD, official splits)

All numbers come from `scripts/evaluate.py` (→ `results/results_test.md`, `results/metrics_test.json`).
Model choices were made on **dev only**. The test set was evaluated once for the final table. It was
regenerated once after an ASR *latency* fix changed the shipped transcripts (see DECISIONS.md section 10).
The pre-fix table is kept in `results/results_test_asr_with_fallback.md`, and it differs by ≤ 0.002 weighted F1.

**Weighted F1** = F1 score per class, averaged with weights equal to each class's frequency (MELD's
standard metric; neutral is 48% of test). **Macro F1** = unweighted average over the 7 classes (rare
classes count equally). ± is the std over 3 training seeds.

| MELD **test** (2,610 utterances) | weighted F1 | macro F1 | accuracy |
|---|---|---|---|
| majority class (always "neutral") | 0.313 | 0.093 | 0.481 |
| text-only, gold transcript | 0.620 ± 0.005 | 0.453 ± 0.003 | 0.623 ± 0.008 |
| audio-only | 0.481 ± 0.007 | 0.308 ± 0.019 | 0.495 ± 0.015 |
| **fused** text+audio, gold transcript | 0.620 ± 0.002 | 0.448 ± 0.001 | 0.621 ± 0.002 |
| text-only, **ASR transcript** | 0.515 ± 0.006 | 0.349 ± 0.008 | 0.536 ± 0.010 |
| **fused** text+audio, **ASR transcript** ← *what the live system does* | **0.530 ± 0.001** | 0.362 ± 0.002 | 0.548 ± 0.003 |
| late fusion (weighted average of probabilities), gold transcript | 0.624 ± 0.009 | 0.452 ± 0.010 | 0.636 ± 0.012 |
| late fusion, ASR transcript | 0.539 ± 0.001 | 0.366 ± 0.003 | 0.566 ± 0.005 |

Whisper small.en word error rate on MELD test: **35.0%**.

**What this says, honestly:**
1. **The ASR gap is the biggest effect in the system.** Swapping gold transcripts for Whisper's costs
   ~10.5 points of weighted F1 (0.620 → 0.515 for text). MELD audio is TV audio with laugh tracks,
   cross-talk, and clip boundaries that often include a neighbouring line.
2. **Audio helps only when the text is noisy.** With gold text, fusion = text (0.620 vs 0.620). With ASR
   text, fusion adds +1.5 points (0.515 → 0.530), consistent across seeds. On dev with gold text it added +0.9.
3. **"Same words, different tone" is not solved.** `scripts/same_words_test.py` takes the 155 test
   utterances whose *exact* text appears with ≥2 different gold emotions ("Hey!", "What?", "Yeah!"…).
   A text model must answer identically for identical text. The fused model scores 0.609 ± 0.015 vs
   text-only 0.613 ± 0.005, i.e. **no measurable gain from tone** on these. Fusion does flip individual
   clips the right way (145 fixed vs 139 broken relative to text-only, seed 0), but the tone signal from a
   frozen 95M speech encoder on MELD's audio is too weak to beat words on aggregate.
4. **Late fusion is as good as the learned MLP fusion, or slightly better.** It was tied on dev (0.612 vs
   0.610). On test with gold text it's within noise (0.624 vs 0.620). On test with ASR text, late fusion is
   slightly better (0.539 vs 0.530, beyond the seed std). The MLP was chosen on dev, before the test run,
   and it gives per-modality probabilities from one model, so it stays. I didn't switch because of a test
   result. The honest takeaway is that a learned fusion layer buys nothing over averaging probabilities here.
5. **The failure mode is "everything collapses to neutral".** In the main confusion matrix
   (`results/confusion_test_fused_text_audio_asr_transcript.png`), 34–46% of every emotion class is
   predicted as neutral. Fear (50 test clips) is nearly undetected (F1 0.12).

Per-class F1 (seed 0, test):

| model | anger | disgust | fear | joy | neutral | sadness | surprise |
|---|---|---|---|---|---|---|---|
| text-only (gold) | 0.47 | 0.25 | 0.27 | 0.58 | 0.78 | 0.33 | 0.53 |
| audio-only | 0.40 | 0.12 | 0.10 | 0.36 | 0.66 | 0.31 | 0.32 |
| fused (gold) | 0.46 | 0.18 | 0.22 | 0.57 | 0.77 | 0.38 | 0.55 |
| fused (ASR) | 0.35 | 0.15 | 0.12 | 0.44 | 0.69 | 0.37 | 0.44 |

For reference (not a like-for-like comparison): published MELD systems that fine-tune encoders and use
dialogue context report weighted F1 in the mid-0.60s with gold text. Our frozen-encoder, no-context,
gold-text 0.620 is in that neighbourhood. Our deployable ASR number (0.530) is the one that matters for the robot.

## Real-time: definition and measurements

**Definition (turn-based).** Audio streams in as 32 ms chunks. Silero VAD declares *end of speech* (EOS)
after 500 ms of silence. From EOS we target **emotion tags (JSON) ≤ ~300 ms + ASR** and **first reply
token ≤ 1 s**.

Measured by `scripts/benchmark_latency.py --n 100` on 100 random MELD test clips (median 2.2 s long),
full live path including the LLM (→ `results/latency_summary.json`):

| stage (ms) | p50 | p95 | max |
|---|---:|---:|---:|
| VAD compute per 32 ms chunk | 0.1 | 0.1 | 0.4 |
| ASR (Whisper small.en, MPS) | 239 | 297 | 327 |
| emotion (RoBERTa + WavLM + head + tracker) | 50 | 84 | 139 |
| **EOS → emotion tags** | **288** | **380** | 449 |
| LLM first token | 86 | 104 | 113 |
| **EOS → first reply token** | **378** | **484** | 549 |
| LLM full reply (≤ 60 tokens) | 174 | 226 | 305 |

p50/p95 = median / 95th percentile. **Plus the fixed 500 ms VAD silence wait** before EOS, so from the
moment the person actually stops talking the first token arrives after ~0.88 s (p50) / ~0.98 s (p95).
Targets met: emotion stage ≤ 300 ms ✔ (p95 84 ms); first token ≤ 1 s from EOS ✔, and also ✔ including the
silence wait, but with almost no margin at p95.

**Hardware & resources** (MacBook Pro, Apple M4 Pro, 24 GB unified memory, macOS 15.5):

| | |
|---|---|
| model load + warm-up | 5.9 s |
| process RSS (resident memory) after load / peak | 1.5 GB / 1.6 GB |
| PyTorch MPS allocations (Whisper is CTranslate2, not in this) | 3.5 GB (driver-level, includes allocator cache) |
| MLX peak (Qwen3-1.7B 4-bit + KV cache) | 1.3 GB |
| model weight files on disk | Whisper small.en (CTranslate2) 484 MB, RoBERTa 499 MB, WavLM ~378 MB, Qwen3-1.7B-4bit 968 MB, head 2.1 MB |

Apple's unified memory means GPU allocations don't fully show up in RSS, which is why they're listed separately.

## Parameter budget (≤ 6B, every model on the inference path)

From `scripts/count_params.py` (→ `results/param_budget.md`):

| stage | model | parameters | how counted |
|---|---|---:|---|
| VAD | Silero VAD | 0.46M | loaded, `p.numel()` |
| ASR | Whisper small.en | 241.7M | safetensors header of the original checkpoint (we run a CTranslate2 conversion) |
| text encoder | RoBERTa-base (no pooler) | 124.1M | loaded, `p.numel()` |
| audio encoder | WavLM-base-plus | 94.4M | loaded, `p.numel()` |
| fusion head | ours | 0.53M | loaded, `p.numel()` |
| LLM | Qwen3-1.7B (run as MLX 4-bit) | 2,031.7M | original safetensors header. Conservative: the tied embedding is stored twice; 1,720.6M unique weights |
| **total (core)** | | **2.493B** | **41.5% of the 6B cap** |
| face detector (vision ext.) | YuNet | 0.053M | ONNX initializers, summed |
| face expression (vision ext.) | `trpakov/vit-face-expression` (ViT-base) | 85.8M | loaded, `p.numel()` |
| **total with vision** | | **2.579B** | **43.0% of the cap** |

The Phase-1 fallback classifier (j-hartmann DistilRoBERTa, 82M) is **not loaded** once `models/fused.pt`
exists. Even if it were, the total would be 2.58B.

## Qualitative examples

Picked **by rule** from test predictions (not hand-picked) by `scripts/qualitative_examples.py`, then run
through the full pipeline. Full text: `results/qualitative_examples.md`.

| case | clip | gold | text head / audio head → fused | Lumi's reply |
|---|---|---|---|---|
| tone overrides words | Rachel: "I guess I just figured that somewhere down the road, we would be on again." | sadness | neutral / sadness → **sadness** (0.64) | "Hmm, that feels sad. Maybe we'll be on again someday." |
| tone overrides words | Gunther: "Do you remember when you first came here, how you spent two weeks getting trained…" | neutral | surprise / neutral → **neutral** (0.59) | "I remember when I first came here. It felt like two weeks, but I was just learning." ← *grounding failure: the LLM answers as if it were the speaker* |
| correct, confident | Phoebe: "Oh that's so exciting!" | joy | joy / joy → **joy** (0.99) | "Oh, that's so exciting! I'm glad you said that. …" |
| low confidence | Bonnie: "Y'know what, I should do it." (ASR heard: "I **can't** do it.") | joy | sadness / surprise → anger (**0.18**, below 0.45) | "Hmm, that sounds tough. What's the thing you're trying to do? …" ← neutral/curious, as designed, but the ASR error flipped the meaning |
| failure | Ross: "Yeah." (ASR heard: "Mm.") | sadness | neutral / joy → **neutral** (0.96) | "Mmm... maybe they're thinking about something. …" ← confidently wrong on a one-word line |

## Vision extension: face as a third modality

```
video frame ─► YuNet face detector (53k params) ─► largest face ─► ViT facial-expression model (85.8M, FER2013)
                                                                      │  7 probabilities, same labels as MELD
text+audio probabilities ──────────────────────────────────────────► p = (1 − w)·p_text_audio + w·p_face
```

- **Live:** a background thread scores the webcam at ~4 frames/s. When an utterance ends, the face
  probabilities are averaged over the time the person was speaking (`run_live.py --vision`). Because it
  runs in parallel, it adds no waiting to the turn itself. The emotion stage did slow down somewhat
  (56 → 104 ms on one simulated clip) from GPU contention.
- **Off-the-shelf, not trained here.** The only fitted number is the fusion weight `w`, chosen on MELD
  dev (`w = 0.1` for the deployed head, saved in `models/vision_fusion.json`).
- The JSON gains `per_modality.face` and a `vision` block. The LLM is told when the face disagrees with
  the fused emotion (e.g. smiling while saying something sad).

Measured on MELD by `scripts/evaluate_vision.py` (→ `results/results_vision.md`). A face was found in 99.1% of clips:

| MELD test | weighted F1 | macro F1 |
|---|---|---|
| face only | 0.267 | 0.158 |
| (always "neutral", for reference) | 0.313 | 0.093 |
| text + audio, ASR transcript | 0.530 ± 0.001 | 0.362 ± 0.002 |
| **text + audio + face**, ASR transcript | **0.533 ± 0.002** | 0.365 ± 0.003 |
| text + audio, gold transcript | 0.620 ± 0.002 | 0.448 ± 0.001 |
| text + audio + face, gold transcript | 0.623 ± 0.005 | 0.453 ± 0.006 |

**Honest reading:** the face model alone is *worse than always saying "neutral"*. It almost never
predicts neutral, disgust, fear or surprise correctly (per-class F1 0.35 / 0.00 / 0.00 / 0.01). With a
small weight it adds ~+0.3 F1. That's consistent across the 3 seeds, but tiny. There are two reasons:
1. **Domain gap.** FER2013 is 48×48 grayscale web photos, while MELD is lit, edited sitcom footage.
2. **Speaker attribution.** "Largest face" is often the listener, or an over-the-shoulder shot. A real
   desk robot looks at one person facing it, so the live webcam setting is closer to what this model
   was trained on than MELD is. That's plausible but **not measured here**: I couldn't test the webcam,
   because camera access is blocked for the agent process that built this.

## Done vs intentionally left out

**Done:** streaming VAD → ASR → text+audio fusion → state tracker → streamed LLM reply, all local; JSON
state per turn with per-modality probabilities and a robot expression tag; live mic demo and file demo
(including any MELD clip by id); MELD pipeline with cached embeddings; text / audio / fused / late-fusion
heads with 3 seeds; gold vs ASR comparison; same-words-different-tone test; latency + memory benchmark;
parameter-count script; 5 tests. **Extension:** vision (face expression) as a third
late-fused modality in both demos, measured on MELD dev/test.

**Intentionally left out (and why):**
- **Fine-tuning the encoders.** It would likely gain a few F1 points but costs hours per run. Frozen
  encoders made 3-seed comparisons and sweeps possible inside the timebox.
- **Dialogue context in the emotion model** (previous utterances). Known to help on MELD, but adds a
  history-dependent feature path. The LLM does get the last 3 turns.
- **Training on ASR transcripts** to close the train/deploy mismatch. Probably the single best next step
  (see Limitations), but it needs Whisper over 10k training clips (~45 min) plus a dev-set rerun.
- **RL.** No reward signal worth optimizing exists without user studies. A measured before/after was
  required, and there was nothing honest to measure.
- **Fine-tuning the face model on MELD frames.** It would need speaker-face attribution (who is
  talking in each shot), which MELD doesn't provide.
- **Text-to-speech output.** The robot "speaks" text in the terminal.

## Limitations

- **The live mic demo was tested only for startup and mic capture on this machine,** not with a real
  conversation. Its streaming code path *is* exercised end to end by `run_live.py --simulate`, which feeds
  a file in real time through the same VAD/segmenter.
- MELD is scripted TV with a laugh track. A desk robot hears one person in a room, at a different
  distance and noise level, so the reported F1 is not a deployment accuracy estimate.
- Train/deploy mismatch: heads are trained on gold transcripts but deployed on ASR transcripts (35% WER).
- The tone signal is weak (point 3 above). The "trust the tone" hint to the LLM only fires when audio is
  ≥ 0.45 confident and disagrees with text.
- Confidence values are softmax outputs and are **not calibrated** (e.g. 0.96 on a wrong "Yeah.").
- Small-LLM grounding failures happen (the Gunther example). Replies are sampled (temperature 0.7) with a
  fixed seed, so the README examples are reproducible.
- With `--vision` on macOS you may see `objc: Class AVFFrameReceiver is implemented in both …`.
  OpenCV and PyAV (used by faster-whisper) each bundle FFmpeg's camera code. It's printed by the macOS
  runtime, so Python can't silence it, and it's harmless in practice.
- The webcam path (`--vision` without `--simulate`) was not tested by me (camera permission). The same
  code runs on video files in `--simulate` mode, which was tested.
- Whisper's MPS backend in CTranslate2 is labelled experimental. There is an automatic CPU fallback
  (~950 ms instead of ~265 ms for small.en).

## External components

| component | what / where | params | license | effect on results |
|---|---|---|---|---|
| MELD | Poria et al. 2019, `web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz` | - | GPL-3.0 | all training/eval data |
| Silero VAD | `silero-vad` 6.2.3 (pip) | 0.46M | MIT | defines utterance boundaries and EOS for latency |
| Whisper small.en | OpenAI, run via `faster-whisper` 1.2.1 / CTranslate2 4.9 (`Systran/faster-whisper-small.en`) | 241.7M | MIT | 35% WER on MELD, the largest accuracy loss |
| RoBERTa-base | `FacebookAI/roberta-base` | 124.1M | MIT | frozen text features |
| WavLM-base-plus | `microsoft/wavlm-base-plus` | 94.4M | see model card (UniLM repo is MIT) | frozen audio features |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B`, 4-bit MLX conversion `mlx-community/Qwen3-1.7B-4bit`, via `mlx-lm` 0.32 | 2.03B | Apache-2.0 | reply text only, does not affect emotion metrics |
| YuNet face detector | OpenCV Zoo, `opencv/face_detection_yunet` (run via `cv2.FaceDetectorYN`) | 0.053M | MIT | vision ext.: finds the face (99.1% of MELD clips) |
| vit-face-expression | `trpakov/vit-face-expression`, ViT-base fine-tuned on FER2013 | 85.8M | see model card | vision ext.: +0.3 F1 at w = 0.1 |
| emotion-english-distilroberta-base | `j-hartmann/...`, Phase-1 fallback only | 82M | see model card | none on reported results (trained partly on MELD → never evaluated on it) |

Licences are as stated on each project's page at the time of writing. Verify them before any commercial use.

## Repo map

```
run_file.py, run_live.py      demos
src/  config, audio_io, vad, asr, encoders, fusion, state_tracker, responder, pipeline, meld, data,
      pretrained_text, vision (extension), quiet (log silencing)
scripts/  download_meld.sh, extract_audio, cache_embeddings, train_head, sweep_fused.sh, evaluate,
          same_words_test, qualitative_examples, benchmark_latency, count_params,
          cache_vision, evaluate_vision
models/   trained heads (text/audio/fused × 3 seeds) + vision_fusion.json
results/  every number in this README (JSON/MD/PNG), generated by the scripts above
tests/    pytest: state tracker, head shapes, VAD splitting, pipeline JSON schema, vision no-face case
DECISIONS.md   options → choice → evidence → trade-off for each design decision
PROGRESS.md    status log
```

