"""Frozen pretrained encoders that turn text / audio into embeddings.

An *embedding* is a fixed-length vector summarizing the input. We take the
hidden states of *every* transformer layer and average them over time
(mean pooling), giving an array of shape [num_layers, 768]. The trained head
then learns how much to trust each layer. Different layers capture different
information (lower layers in speech models carry more prosody/tone, higher
layers carry more content), and that's the standard SUPERB-benchmark recipe.

The encoders stay frozen (never trained), which means:
  * we can compute embeddings once and cache them, so training takes seconds;
  * the live pipeline uses exactly the same features as training.
"""
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # hide load reports for heads we deliberately don't use

from src.config import AUDIO_ENCODER, MAX_AUDIO_SECONDS, SAMPLE_RATE, TEXT_ENCODER


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _masked_mean(hidden: tuple, mask: torch.Tensor) -> np.ndarray:
    """hidden: tuple of [B, T, D] per layer; mask: [B, T] -> [B, L, D] numpy."""
    h = torch.stack(hidden, dim=1)                       # [B, L, T, D]
    m = mask[:, None, :, None].to(h.dtype)               # [B, 1, T, 1]
    pooled = (h * m).sum(2) / m.sum(2).clamp(min=1)
    return pooled.float().cpu().numpy()


class TextEncoder:
    def __init__(self, name: str = TEXT_ENCODER, device: str | None = None):
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        # add_pooling_layer=False: RoBERTa's 'pooler' layer is untrained in this checkpoint
        # and we never use it (we pool hidden states ourselves), so don't load or count it.
        self.model = AutoModel.from_pretrained(name, add_pooling_layer=False).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tok(texts, padding=True, truncation=True, max_length=96,
                         return_tensors="pt").to(self.device)
        out = self.model(**batch, output_hidden_states=True)
        return _masked_mean(out.hidden_states, batch["attention_mask"])


class AudioEncoder:
    def __init__(self, name: str = AUDIO_ENCODER, device: str | None = None):
        self.device = device or pick_device()
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """One utterance at a time (no padding, so no padding artefacts)."""
        audio = audio[: int(MAX_AUDIO_SECONDS * SAMPLE_RATE)]
        if len(audio) < SAMPLE_RATE // 4:                # pad very short clips to 0.25 s
            audio = np.pad(audio, (0, SAMPLE_RATE // 4 - len(audio)))
        x = self.fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = x["input_values"].to(self.device)
        out = self.model(x, output_hidden_states=True)
        mask = torch.ones(out.hidden_states[0].shape[:2], device=self.device)
        return _masked_mean(out.hidden_states, mask)[0]  # [L, D]
