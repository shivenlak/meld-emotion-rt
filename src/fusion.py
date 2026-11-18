"""The only part we train: a small head on top of frozen embeddings.

Per modality (text, audio):
    layer embeddings [L, 768]
      -> learned softmax weights over the L layers -> one 768-d vector
      -> LayerNorm -> Linear(768, 256) -> ReLU -> Dropout   ("projection")
      -> Linear(256, 7)                                      ("aux" classifier)

Fused model: concatenate the two 256-d projections -> small MLP -> 7.

The outputs are *logits*: one raw score per emotion. softmax(logits) turns
them into probabilities that sum to 1. The aux classifiers are trained jointly
with the fused one, so a single fused model also gives per-modality
probabilities (shown in the JSON as `per_modality`).

*Modality dropout*: during training we sometimes zero one modality's
projection, so the fused classifier can't lean entirely on text (the
stronger signal) and has to learn to use audio too.
"""
import torch
import torch.nn as nn

from src.config import EMOTIONS

N_CLASSES = len(EMOTIONS)


class ModalityBranch(nn.Module):
    def __init__(self, n_layers: int, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))  # start as a plain average
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.aux = nn.Linear(hidden, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, D]
        w = torch.softmax(self.layer_logits, dim=0)
        return self.proj((x * w[None, :, None]).sum(1))  # [B, hidden]


class EmotionHead(nn.Module):
    def __init__(self, modalities: list[str], n_layers: dict[str, int],
                 hidden: int = 256, dropout: float = 0.3, modality_dropout: float = 0.0):
        super().__init__()
        self.modalities = list(modalities)
        self.modality_dropout = modality_dropout
        self.branches = nn.ModuleDict({m: ModalityBranch(n_layers[m], hidden=hidden, dropout=dropout)
                                       for m in self.modalities})
        if len(self.modalities) > 1:
            self.fused = nn.Sequential(nn.Linear(hidden * len(self.modalities), hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = {m: self.branches[m](feats[m]) for m in self.modalities}
        out = {m: self.branches[m].aux(z[m]) for m in self.modalities}
        if len(self.modalities) == 1:
            out["fused"] = out[self.modalities[0]]
            return out
        parts = []
        for m in self.modalities:
            h = z[m]
            if self.training and self.modality_dropout > 0:
                keep = (torch.rand(h.shape[0], 1, device=h.device) > self.modality_dropout).to(h.dtype)
                h = h * keep
            parts.append(h)
        out["fused"] = self.fused(torch.cat(parts, dim=1))
        return out


def save_head(model: EmotionHead, path, config: dict):
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_head(path, device: str = "cpu") -> EmotionHead:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = EmotionHead(c["modalities"], c["n_layers"], hidden=c["hidden"], dropout=c["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.config = c
    return model.to(device).eval()
