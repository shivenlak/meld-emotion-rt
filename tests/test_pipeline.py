"""Fast-ish tests: `pytest -q` (~30 s, loads ASR + encoders once; skips the LLM)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_io import load_audio  # noqa: E402
from src.config import EMOTIONS  # noqa: E402
from src.fusion import EmotionHead  # noqa: E402
from src.state_tracker import EmotionStateTracker  # noqa: E402

SAMPLE = ROOT / "samples" / "two_turns_tts.wav"


def test_state_tracker_ema():
    tr = EmotionStateTracker(alpha=0.7)
    joy, sad = np.eye(7)[EMOTIONS.index("joy")], np.eye(7)[EMOTIONS.index("sadness")]
    assert tr.update(joy)["label"] == "joy"
    s = tr.update(sad)  # 0.7 * sad + 0.3 * joy -> sadness wins
    assert s["label"] == "sadness" and abs(s["confidence"] - 0.7) < 1e-6


def test_fusion_head_shapes():
    head = EmotionHead(["text", "audio"], {"text": 13, "audio": 13}, modality_dropout=0.5)
    out = head({"text": torch.randn(4, 13, 768), "audio": torch.randn(4, 13, 768)})
    assert set(out) == {"text", "audio", "fused"}
    assert all(v.shape == (4, 7) for v in out.values())


def test_vad_splits_two_turns():
    from src.vad import split_file
    segs = split_file(load_audio(SAMPLE))
    assert len(segs) == 2


@pytest.fixture(scope="module")
def pipe():
    from src.pipeline import Pipeline
    return Pipeline(load_llm=False)


def test_pipeline_output_schema(pipe):
    from src.vad import split_file
    seg = split_file(load_audio(SAMPLE))[0]
    state = pipe.process(seg["audio"], t_end_speech_ms=seg["end_ms"])
    json.dumps(state)  # must be JSON-serialisable
    required = {"turn_id", "transcript", "emotion", "confidence", "probs", "per_modality",
                "smoothed_state", "expression", "latency_ms"}
    assert required <= set(state)
    assert state["emotion"] in EMOTIONS and state["smoothed_state"] in EMOTIONS
    assert abs(sum(state["probs"].values()) - 1) < 0.01
    assert 0 <= state["confidence"] <= 1
    assert "job" in state["transcript"].lower()
    assert {"asr", "emotion"} <= set(state["latency_ms"])


def test_vision_no_face_returns_none():
    """Vision extension: a blank frame has no face -> (None, 0), and the label order maps to MELD's."""
    from src.vision import FER_TO_MELD, FaceEmotion
    fe = FaceEmotion()
    probs, n = fe.predict_frames([np.zeros((480, 640, 3), np.uint8)])
    assert probs is None and n == 0
    assert sorted(FER_TO_MELD.values()) == sorted(EMOTIONS)
    assert sorted(fe.order) == list(range(7))
