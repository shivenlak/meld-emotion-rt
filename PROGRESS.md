# PROGRESS

## Status: core (phases 0–6) complete. Vision extension (phase 7) done. Published on GitHub.

### Done
- Env: Python 3.11 venv, PyTorch 2.15 MPS, ffmpeg 8.1, M4 Pro 24 GB.
- MELD downloaded + extracted (archive deleted), 13,706 / 13,708 clips → 16 kHz wav (2 known-broken clips:
  train dia125_utt3, dev dia110_utt7).
- Pipeline: Silero VAD → Whisper small.en (MPS) → RoBERTa + WavLM (frozen) → fusion head → EMA tracker →
  Qwen3-1.7B 4-bit MLX. `run_file.py`, `run_live.py` (+ `--simulate`).
- Trained text / audio / fused heads, 3 seeds each; dev sweep; one test evaluation (regenerated once after the
  ASR latency fix, see DECISIONS.md section 10).
- Results: test weighted F1: text(gold) 0.620, audio 0.481, fused(gold) 0.620, fused(ASR) 0.530.
  Latency EOS→first token p50 378 ms / p95 484 ms (+500 ms VAD wait). 2.493B params.
- README, DECISIONS, 5 passing tests.
- Vision extension (2026-11-18): YuNet + ViT FER, late fusion w = 0.1 (dev). Test: face-only 0.267;
  text+audio+face (ASR) 0.533 vs 0.530 without. `--vision` flag in both demos. Library log noise hidden
  by default (`--verbose` to show).

### Not verified
- Live mic with a real speaker (only startup + mic capture checked by me). Try: `python run_live.py`.
- Webcam (`--vision` without `--simulate`): camera permission was blocked for the agent; tested only on MELD video via `--simulate`.
- Fresh-copy test DONE (2026-11-18): git-tracked files only → new venv from requirements.txt → pytest (4 pass),
  run_file.py on the sample (trained_fusion), count_params. Caveat: shared ~/.cache/huggingface, so first-time
  model downloads were not re-tested, and the MELD reproduction steps were not re-run from scratch.

### Next (in priority order)
1. Train heads on Whisper transcripts of train/dev (closes the gold→ASR mismatch; ~1 h of ASR).
2. Switch to late fusion (simpler, ≥ MLP on test). Re-validate on dev.
3. Dialogue context for the text encoder (previous utterance).
4. Vision: active-speaker detection (pick the face whose mouth moves) instead of the largest face.
