"""Vision extension: face -> facial-expression probabilities (third modality).

    frame (BGR, any size)
      -> YuNet face detector (OpenCV's small CNN, ~75k params) -> largest face box
      -> crop with a margin -> ViT facial-expression classifier
         (trpakov/vit-face-expression, ViT-base, trained on FER2013)
      -> 7 probabilities in MELD's label order

The ViT's 7 classes (angry, disgust, fear, happy, neutral, sad, surprise) map
one-to-one onto MELD's labels, so no retraining is needed. Vision joins the
system by *late fusion*: its probabilities are averaged with the text+audio
probabilities using a weight chosen on MELD dev (scripts/evaluate_vision.py).

Off-the-shelf caveat: FER2013 is 48x48 grayscale posed/web faces; MELD frames
are TV shots where the largest face is not always the speaker.
"""
import threading
import time

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.config import EMOTIONS, FACE_DETECTOR_FILE, FACE_DETECTOR_REPO, FACE_EMOTION_MODEL
from src.encoders import pick_device

# FER label -> MELD label
FER_TO_MELD = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "joy",
               "neutral": "neutral", "sad": "sadness", "surprise": "surprise"}


class FaceEmotion:
    def __init__(self, min_face_px: int = 40):
        self.device = pick_device()
        onnx = hf_hub_download(FACE_DETECTOR_REPO, FACE_DETECTOR_FILE)
        self.detector = cv2.FaceDetectorYN.create(onnx, "", (320, 320), score_threshold=0.7)
        self.proc = AutoImageProcessor.from_pretrained(FACE_EMOTION_MODEL)
        self.model = AutoModelForImageClassification.from_pretrained(FACE_EMOTION_MODEL).to(self.device).eval()
        id2label = self.model.config.id2label
        meld_of = {i: FER_TO_MELD[l.lower()] for i, l in id2label.items()}
        self.order = [next(i for i, m in meld_of.items() if m == e) for e in EMOTIONS]
        self.min_face_px = min_face_px

    def largest_face(self, frame: np.ndarray):
        """Returns the face crop (RGB) of the largest detected face, or None."""
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)  # detect on a downscaled copy: faster, same boxes after rescaling
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = (faces[np.argmax(faces[:, 2] * faces[:, 3]), :4] / scale).astype(int)
        if min(fw, fh) < self.min_face_px:
            return None
        m = int(0.15 * max(fw, fh))  # margin: FER models expect a bit of context around the face
        crop = frame[max(0, y - m):y + fh + m, max(0, x - m):x + fw + m]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size else None

    @torch.no_grad()
    def predict_faces(self, faces: list[np.ndarray]) -> np.ndarray:
        x = self.proc(images=faces, return_tensors="pt")["pixel_values"].to(self.device)
        p = torch.softmax(self.model(pixel_values=x).logits, -1).cpu().numpy()
        return p[:, self.order]

    def predict_frames(self, frames: list[np.ndarray]) -> tuple[np.ndarray | None, int]:
        """(mean probabilities over frames that contain a face, number of such frames).
        Probabilities are None if no frame had a face."""
        faces = [f for f in (self.largest_face(fr) for fr in frames) if f is not None]
        return (self.predict_faces(faces).mean(0) if faces else None), len(faces)


def video_frames(path: str, n: int = 5, start_s: float = 0.0, end_s: float | None = None) -> list[np.ndarray]:
    """n evenly spaced frames from a video file (optionally between start_s and end_s)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = int(start_s * fps)
    b = min(total - 1, int(end_s * fps)) if end_s else total - 1
    frames = []
    for idx in np.linspace(a, max(a, b), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


class FaceWatcher:
    """Background thread for the live demo: keeps scoring faces at ~`fps` and
    stores (timestamp, probabilities). At the end of an utterance the pipeline
    asks for the mean over the utterance's time window.

    Source is a webcam index (0) or a video file (for --simulate), read in real time."""

    def __init__(self, face: FaceEmotion, source=0, fps: float = 4.0):
        self.face, self.source, self.period = face, source, 1.0 / fps
        self.history: list[tuple[float, np.ndarray]] = []
        self.lock = threading.Lock()
        self.running = True
        self.error = None
        self.frames_seen = 0
        self.t0 = time.perf_counter()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"could not open video source {self.source!r} (camera permission?)"
            return
        is_file = not isinstance(self.source, int)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while self.running:
            if is_file:  # jump to the frame matching wall-clock time, like a live camera would
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((time.perf_counter() - self.t0) * fps))
            ok, frame = cap.read()
            if not ok:
                break
            self.frames_seen += 1
            p, _ = self.face.predict_frames([frame])
            if p is not None:
                with self.lock:
                    self.history.append((time.perf_counter(), p))
                    self.history = self.history[-200:]
            time.sleep(self.period)
        cap.release()

    def window_probs(self, t_start: float, t_end: float) -> tuple[np.ndarray | None, int]:
        with self.lock:
            ps = [p for t, p in self.history if t_start <= t <= t_end]
        return (np.mean(ps, 0) if ps else None), len(ps)

    def stop(self):
        self.running = False
