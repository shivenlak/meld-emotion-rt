"""Speech recognition (ASR) with faster-whisper.

faster-whisper runs OpenAI's Whisper through CTranslate2, an optimized
inference engine. CTranslate2 4.9 has an *experimental* Apple-GPU (MPS)
backend. On a 3 s clip, small.en takes ~264 ms on MPS vs ~950 ms on CPU int8
(measured in this repo, see DECISIONS.md), with an identical transcript. We use MPS and
fall back to CPU if MPS fails to initialise.
"""
import numpy as np
from faster_whisper import WhisperModel

from src.config import ASR_MODEL


class ASR:
    def __init__(self, model_name: str = ASR_MODEL):
        try:
            self.model = WhisperModel(model_name, device="mps", compute_type="float16")
            self.device = "mps"
        except Exception:
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.device = "cpu"

    def transcribe(self, audio: np.ndarray) -> str:
        # beam_size=1 (greedy decoding) is the fastest. vad_filter is off
        # because our own VAD has already cut the utterance.
        # temperature=0.0 disables faster-whisper's "temperature fallback": by
        # default, a decode that looks bad is retried up to 5 more times at
        # higher temperature. We measured that turning a 4 s clip into 5-6 s of
        # ASR (and a hallucinated "Okay. Okay. Okay..." loop). One decode plus a
        # token cap (~10 tokens per second of audio) gives a hard latency bound.
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0, without_timestamps=True,
            max_new_tokens=min(int(10 * len(audio) / 16000) + 10, 400),  # Whisper max is 448 per 30 s window
        )
        return " ".join(s.text.strip() for s in segments).strip()
