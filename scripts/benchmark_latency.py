"""Latency + resource benchmark of the full live path on real MELD test clips.

    python scripts/benchmark_latency.py --n 100

For each clip: replay it through the streaming VAD (chunk by chunk, as the
mic would deliver it, but not paced in real time), then ASR -> encoders ->
fusion -> state tracker -> LLM reply. Records per-stage latency, then reports
p50 / p95 (50th and 95th percentile: "typical" and "bad-case" latency).

Memory: macOS uses unified memory, so GPU allocations don't fully show in the
process RSS. We report process peak RSS plus PyTorch-MPS and MLX GPU
allocations separately.
"""
import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.audio_io import quiet_noise  # noqa: E402
from src.config import RESULTS_DIR, SAMPLE_RATE  # noqa: E402
from src.meld import load_split  # noqa: E402
from src.vad import CHUNK, StreamingSegmenter  # noqa: E402


class PeakRSS:
    def __init__(self):
        self.peak, self.stop = 0, False
        self.proc = psutil.Process()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.05)


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mem = PeakRSS()
    rss_before = psutil.Process().memory_info().rss
    from src.pipeline import Pipeline
    pipe = Pipeline(load_llm=True)
    rss_loaded = psutil.Process().memory_info().rss

    df = load_split("test")
    df = df[df["wav"].map(lambda p: Path(p).exists())].sample(n=args.n, random_state=args.seed)
    rows = []
    for uid, path in zip(df["uid"], df["wav"]):
        audio, _ = sf.read(path, dtype="float32")
        audio = np.concatenate([audio, quiet_noise(SAMPLE_RATE)])
        seg, vad_times, utt = StreamingSegmenter(min_silence_ms=500), [], None
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            t = time.perf_counter()
            out = seg.feed(audio[i:i + CHUNK])
            vad_times.append((time.perf_counter() - t) * 1000)
            if out is not None and utt is None:
                utt = out
        if utt is None:  # VAD heard no speech (laugh track / very short clip): use the whole clip
            utt = audio[:-SAMPLE_RATE]
        pipe.tracker.reset()
        pipe.history.clear()
        state = pipe.process(utt, vad_ms=float(np.mean(vad_times)))
        for _ in pipe.respond(state):
            pass
        lat = state["latency_ms"]
        lat["tags_after_eos"] = lat["asr"] + lat["emotion"]          # end-of-speech -> JSON tags
        lat["first_token_after_eos"] = lat["tags_after_eos"] + lat["llm_first_token"]
        rows.append({"uid": uid, "audio_seconds": state["audio_seconds"], **lat})
    mem.stop = True

    import mlx.core as mx
    import torch
    stages = ["vad", "asr", "emotion", "llm_first_token", "llm_total", "tags_after_eos", "first_token_after_eos"]
    summary = {
        "n_clips": len(rows),
        "hardware": {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                            text=True).stdout.strip(),
                     "ram_gb": round(psutil.virtual_memory().total / 2**30), "os": platform.platform(),
                     "asr_device": pipe.asr.device},
        "audio_seconds_p50": pct([r["audio_seconds"] for r in rows], 50),
        "latency_ms": {s: {"p50": pct([r[s] for r in rows], 50), "p95": pct([r[s] for r in rows], 95),
                           "max": pct([r[s] for r in rows], 100)} for s in stages},
        "vad_wait_ms": 500,
        "memory": {"rss_before_load_mb": round(rss_before / 2**20), "rss_after_load_mb": round(rss_loaded / 2**20),
                   "rss_peak_mb": round(mem.peak / 2**20),
                   "torch_mps_driver_mb": round(torch.mps.driver_allocated_memory() / 2**20),
                   "mlx_peak_mb": round(mx.get_peak_memory() / 2**20)},
        "model_load_seconds": pipe.load_seconds,
    }
    print(json.dumps(summary, indent=1))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latency_summary.json").write_text(json.dumps(summary, indent=1))
    (RESULTS_DIR / "latency_per_clip.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
