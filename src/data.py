"""Load cached embeddings + compute metrics. Shared by train_head / evaluate."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import CACHE_DIR, EMOTIONS


def load_feats(split: str, modalities: list[str], text_source: str = "gold"):
    """Returns ({modality: float32 tensor [N, L, D]}, labels [N], uids).

    text_source="asr" swaps gold transcripts for Whisper transcripts (test only).
    """
    feats, labels, uids = {}, None, None
    for m in modalities:
        name = f"{split}_asr_text" if (m == "text" and text_source == "asr") else f"{split}_{m}"
        z = np.load(CACHE_DIR / f"{name}.npz")
        feats[m] = torch.from_numpy(z["emb"].astype(np.float32))
        if labels is None:
            labels, uids = z["label"], z["uid"]
        assert (z["uid"] == uids).all(), "caches are not aligned"
    return feats, torch.from_numpy(labels).long(), uids


def metrics(y_true, y_pred) -> dict:
    """Weighted F1 = per-class F1 averaged with weights = class frequency
    (MELD's standard metric, dominated by 'neutral'). Macro F1 = plain
    average over the 7 classes, so rare classes (fear, disgust) count equally."""
    labels = list(range(len(EMOTIONS)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class_f1": {e: round(float(f), 4) for e, f in zip(EMOTIONS, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
