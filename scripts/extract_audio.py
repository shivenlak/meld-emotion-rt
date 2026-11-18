"""MELD mp4 -> 16 kHz mono wav (one per utterance), using ffmpeg in parallel.

    python scripts/extract_audio.py

Skips files already done. Logs clips that are missing/broken to
results/extract_audio_failures.json instead of crashing.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, ROOT  # noqa: E402
from src.meld import SPLITS, load_split  # noqa: E402


def extract(video: str, wav: str) -> str | None:
    if Path(wav).exists() and Path(wav).stat().st_size > 1000:
        return None
    if not Path(video).exists():
        return "missing video"
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", video,
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                       capture_output=True, text=True)
    if r.returncode != 0 or not Path(wav).exists():
        Path(wav).unlink(missing_ok=True)
        # relative paths only: this file is committed, so don't leak the local home directory
        return "ffmpeg failed: " + r.stderr.strip()[-200:].replace(str(ROOT) + "/", "")
    return None


def main():
    failures = {}
    for split in SPLITS:
        df = load_split(split)
        Path(df["wav"].iloc[0]).parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=10) as ex:
            errs = list(ex.map(extract, df["video"], df["wav"]))
        bad = {u: e for u, e in zip(df["uid"], errs) if e}
        failures[split] = bad
        print(f"{split}: {len(df)} utterances, {len(bad)} failed")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "extract_audio_failures.json").write_text(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
