"""Load any audio/video file as 16 kHz mono float32, using ffmpeg.

ffmpeg handles mp4 (MELD clips), wav, m4a, etc. and does the resampling,
so the rest of the code only ever sees one format.
"""
import subprocess

import numpy as np

from src.config import SAMPLE_RATE


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def quiet_noise(n: int, level: float = 1e-3, seed: int = 0) -> np.ndarray:
    """A quiet noise floor (~-60 dBFS) to use as 'silence' when padding files.

    A real microphone never outputs exact zeros, and Whisper behaves badly on
    long runs of digital silence (it hallucinates), so padding with zeros
    would make file-based tests unrealistic."""
    return (np.random.default_rng(seed).standard_normal(n) * level).astype(np.float32)
