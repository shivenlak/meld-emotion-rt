"""Voice activity detection (VAD) with Silero VAD.

VAD answers "is someone speaking in this 32 ms chunk?". We use it to cut a
continuous audio stream into utterances: an utterance ends when we see
`min_silence_ms` of silence after speech. That end-of-speech moment is the
clock start for all our latency numbers.
"""
import numpy as np
import torch
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad

from src.config import SAMPLE_RATE

CHUNK = 512  # Silero needs exactly 512 samples (32 ms) per call at 16 kHz.

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


def split_file(audio: np.ndarray, min_silence_ms: int = 500) -> list[dict]:
    """Offline: return the speech segments of a whole recording, with times."""
    stamps = get_speech_timestamps(
        torch.from_numpy(audio), get_model(), sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
    )
    return [{"start_ms": s["start"] / 16, "end_ms": s["end"] / 16, "audio": audio[s["start"]:s["end"]]}
            for s in stamps]


class StreamingSegmenter:
    """Online: feed 512-sample chunks, get back a full utterance when it ends.

    Used by the live mic demo (and by `--simulate` to replay a file as if it
    were a mic, so the streaming path is testable without a microphone).
    """

    def __init__(self, min_silence_ms: int = 500, pad_ms: int = 100):
        self.it = VADIterator(get_model(), sampling_rate=SAMPLE_RATE,
                              min_silence_duration_ms=min_silence_ms,
                              speech_pad_ms=pad_ms)
        self.buffer: list[np.ndarray] = []
        self.in_speech = False
        # Keep a little audio from before speech start, so first word isn't clipped.
        self.preroll: list[np.ndarray] = []

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        event = self.it(torch.from_numpy(chunk), return_seconds=False)
        if self.in_speech:
            self.buffer.append(chunk)
        else:
            self.preroll = (self.preroll + [chunk])[-4:]
        if event and "start" in event:
            self.in_speech = True
            self.buffer = list(self.preroll)
        if event and "end" in event and self.in_speech:
            self.in_speech = False
            utt = np.concatenate(self.buffer)
            self.buffer = []
            return utt
        return None
