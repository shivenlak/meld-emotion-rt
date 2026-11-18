"""One utterance in -> structured emotion state + streamed reply out.

    audio utterance (already cut by VAD)
      -> ASR transcript
      -> text embedding (RoBERTa) + audio embedding (WavLM)
      -> trained fusion head -> 7 probabilities (+ per-modality probabilities)
      -> state tracker (smoothed mood over turns)
      -> LLM reply (streamed)

If no trained head exists yet (models/fused.pt), it falls back to the
off-the-shelf text classifier, which is the Phase-1 "thin slice".
"""
import time

import numpy as np
import torch

from src.asr import ASR
from src.config import EMOTIONS, MODELS_DIR
from src.state_tracker import EmotionStateTracker

# What a lamp robot could physically do with each emotion. Plain lookup, so
# a motor/LED controller can consume it without parsing text.
EXPRESSION = {
    "anger": "dim_warm_red_slow_lean_back", "disgust": "turn_away_slightly",
    "fear": "lower_head_soft_blue", "joy": "bounce_bright_yellow",
    "neutral": "idle_soft_white", "sadness": "droop_dim_blue",
    "surprise": "pop_up_bright_flash",
}


def _probs_dict(p: np.ndarray) -> dict:
    return {e: round(float(v), 3) for e, v in zip(EMOTIONS, p)}


class Pipeline:
    def __init__(self, head_path=None, load_llm: bool = True, asr_model: str | None = None, vision: bool = False):
        t = time.perf_counter()
        self.asr = ASR(asr_model) if asr_model else ASR()
        head_path = head_path or MODELS_DIR / "fused.pt"
        if head_path.exists():
            from src.encoders import AudioEncoder, TextEncoder
            from src.fusion import load_head
            self.mode = "trained_fusion"
            self.text_enc, self.audio_enc = TextEncoder(), AudioEncoder()
            self.head = load_head(head_path)
        else:
            from src.pretrained_text import PretrainedTextEmotion
            self.mode = "pretrained_text_only"
            self.text_clf = PretrainedTextEmotion()
        # Optional third modality (vision extension): face-expression probabilities,
        # late-fused as p = (1 - w) * p_text_audio + w * p_face, with w chosen on MELD dev.
        self.face = None
        self.vision_weight = 0.0
        if vision:
            import json

            from src.config import VISION_FUSION_FILE
            from src.vision import FaceEmotion
            self.face = FaceEmotion()
            if VISION_FUSION_FILE.exists():
                self.vision_weight = json.loads(VISION_FUSION_FILE.read_text())["w_face"]
        self.responder = None
        if load_llm:
            from src.responder import Responder
            self.responder = Responder()
        self.tracker = EmotionStateTracker()
        self.history: list[tuple[str, str]] = []
        self.turn_id = 0
        self.warmup()
        self.load_seconds = round(time.perf_counter() - t, 1)

    def warmup(self):
        """The first GPU call compiles kernels and is ~10x slower. Do it at load
        time so the first real turn has normal latency. Then reset state."""
        noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        state = self.process(noise)
        state["transcript"] = state["transcript"] or "hello"
        for _ in self.respond(state):
            pass
        self.tracker.reset()
        self.history.clear()
        self.turn_id = 0

    @torch.no_grad()
    def classify(self, audio: np.ndarray, transcript: str) -> tuple[np.ndarray, dict]:
        if self.mode == "pretrained_text_only":
            p = self.text_clf.predict(transcript)
            return p, {"text": _probs_dict(p)}
        txt = torch.from_numpy(self.text_enc.encode([transcript or " "]))
        aud = torch.from_numpy(self.audio_enc.encode(audio))[None]
        out = self.head({"text": txt, "audio": aud})
        probs = {k: torch.softmax(v[0], -1).numpy() for k, v in out.items()}
        return probs["fused"], {m: _probs_dict(probs[m]) for m in ("text", "audio")}

    def process(self, audio: np.ndarray, t_end_speech_ms: float | None = None, vad_ms: float | None = None,
                face_probs: np.ndarray | None = None, face_frames: int = 0) -> dict:
        """Everything up to (not including) the reply. Returns the JSON state.

        face_probs: mean face-expression probabilities over the utterance (vision
        extension), or None if vision is off / no face was seen."""
        lat = {}
        if vad_ms is not None:
            lat["vad"] = round(vad_ms, 1)
        t = time.perf_counter()
        transcript = self.asr.transcribe(audio)
        lat["asr"] = round((time.perf_counter() - t) * 1000, 1)

        t = time.perf_counter()
        p, per_modality = self.classify(audio, transcript)
        if face_probs is not None:
            per_modality["face"] = _probs_dict(face_probs)
            p = (1 - self.vision_weight) * p + self.vision_weight * face_probs
        smoothed = self.tracker.update(p)
        lat["emotion"] = round((time.perf_counter() - t) * 1000, 1)

        self.turn_id += 1
        i = int(p.argmax())
        return {
            "turn_id": self.turn_id,
            "t_end_speech_ms": None if t_end_speech_ms is None else round(t_end_speech_ms),
            "audio_seconds": round(len(audio) / 16000, 2),
            "transcript": transcript,
            "emotion": EMOTIONS[i],
            "confidence": round(float(p[i]), 3),
            "probs": _probs_dict(p),
            "per_modality": per_modality,
            "smoothed_state": smoothed["label"],
            "smoothed_confidence": smoothed["confidence"],
            "expression": EXPRESSION[smoothed["label"]],
            "classifier": self.mode,
            **({"vision": {"face_frames": face_frames, "w_face": self.vision_weight}} if self.face else {}),
            "latency_ms": lat,
        }

    def respond(self, state: dict):
        """Stream the reply. Fills state['reply'] and latency fields as it goes."""
        if self.responder is None:
            return
        pieces = []
        ms = 0.0
        for piece, ms in self.responder.stream(state, self.history):
            if not pieces:
                state["latency_ms"]["llm_first_token"] = round(ms, 1)
            pieces.append(piece)
            yield piece
        state["latency_ms"]["llm_total"] = round(ms, 1)
        state["reply"] = "".join(pieces).strip()
        self.history.append((state["transcript"], state["reply"]))
