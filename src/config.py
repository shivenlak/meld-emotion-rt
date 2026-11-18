"""Single place for model names, labels and paths, so every script agrees."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MELD_DIR = DATA_DIR / "MELD.Raw"          # extracted official release
WAV_DIR = DATA_DIR / "wav"                 # 16 kHz mono wav per utterance
CACHE_DIR = ROOT / "cache"                 # cached embeddings (.npz)
MODELS_DIR = ROOT / "models"               # trained heads
RESULTS_DIR = ROOT / "results"

SAMPLE_RATE = 16_000

# MELD's 7 emotion labels, in a fixed order. Index i in any probability vector
# always means EMOTIONS[i].
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# ---- External models (all run locally) ----
ASR_MODEL = "small.en"                     # faster-whisper (CTranslate2 conversion of openai/whisper-small.en)
ASR_HF_REPO = "openai/whisper-small.en"    # original weights, used only to count parameters
TEXT_ENCODER = "FacebookAI/roberta-base"
AUDIO_ENCODER = "microsoft/wavlm-base-plus"
PRETRAINED_TEXT_EMOTION = "j-hartmann/emotion-english-distilroberta-base"  # Phase-1 fallback only
LLM_MLX = "mlx-community/Qwen3-1.7B-4bit"
LLM_HF_REPO = "Qwen/Qwen3-1.7B"            # original weights, used only to count parameters

# Longest audio we feed the audio encoder. MELD has a few very long clips;
# capping bounds memory/latency, and emotion is usually clear within 15 s.
MAX_AUDIO_SECONDS = 15.0

# ---- Vision extension (optional third modality) ----
FACE_DETECTOR_REPO = "opencv/face_detection_yunet"
FACE_DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
FACE_EMOTION_MODEL = "trpakov/vit-face-expression"
VISION_FUSION_FILE = MODELS_DIR / "vision_fusion.json"   # late-fusion weight chosen on MELD dev
