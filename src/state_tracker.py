"""Smooths the emotion over turns, so the robot doesn't flip mood on every sentence.

Exponential moving average (EMA) over the fused probability vectors:
    state_t = alpha * p_t + (1 - alpha) * state_{t-1}
alpha = 0.7 means the current turn counts most, but a single odd prediction
can't erase the conversation's mood. The per-turn label is still reported
separately; `smoothed_state` is what a robot's slow behaviours (lighting,
posture) would follow.
"""
import numpy as np

from src.config import EMOTIONS


class EmotionStateTracker:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.state: np.ndarray | None = None

    def update(self, probs: np.ndarray) -> dict:
        probs = np.asarray(probs, dtype=np.float64)
        self.state = probs if self.state is None else self.alpha * probs + (1 - self.alpha) * self.state
        i = int(self.state.argmax())
        return {"label": EMOTIONS[i], "confidence": round(float(self.state[i]), 3)}

    def reset(self):
        self.state = None
