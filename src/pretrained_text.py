"""Off-the-shelf text emotion classifier, used ONLY as a fallback before the
fusion head is trained (the "thin slice" of Phase 1).

j-hartmann/emotion-english-distilroberta-base predicts exactly MELD's 7
labels. Its training data includes MELD, so its MELD-test scores would be
inflated by leakage, and we never report it as a result.
"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import EMOTIONS, PRETRAINED_TEXT_EMOTION
from src.encoders import pick_device


class PretrainedTextEmotion:
    def __init__(self, name: str = PRETRAINED_TEXT_EMOTION):
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        id2label = self.model.config.id2label
        # Reorder the model's outputs into our EMOTIONS order.
        self.order = [next(i for i, l in id2label.items() if l == e) for e in EMOTIONS]

    @torch.no_grad()
    def predict(self, text: str) -> np.ndarray:
        x = self.tok(text or " ", return_tensors="pt", truncation=True).to(self.device)
        p = torch.softmax(self.model(**x).logits[0], -1).cpu().numpy()
        return p[self.order]
