"""Read the official MELD CSVs into one clean table per split."""
import pandas as pd

from src.config import EMOTIONS, MELD_DIR, WAV_DIR

SPLITS = ["train", "dev", "test"]
CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
# Folder names inside the official release (inconsistent, but that's the release).
VIDEO_DIR = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}

# MELD's CSVs were saved with Windows-1252 smart quotes that come out as
# mojibake when read as UTF-8 (e.g. "I\x92m"). Map them back to plain ASCII.
_FIX = {"\x92": "'", "\x91": "'", "\x93": '"', "\x94": '"', "\x97": "-", "\x96": "-", "\x85": "...",
        "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€”": "-", "â€“": "-", "â€¦": "...", "\xa0": " "}


def clean_text(s: str) -> str:
    for bad, good in _FIX.items():
        s = s.replace(bad, good)
    return " ".join(s.split())


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(MELD_DIR / CSV[split], encoding="utf-8")
    df["uid"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
    df["text"] = df["Utterance"].astype(str).map(clean_text)
    df["label"] = df["Emotion"].str.lower().map(EMOTIONS.index)
    df["video"] = df["uid"].map(lambda u: str(MELD_DIR / VIDEO_DIR[split] / f"{u}.mp4"))
    df["wav"] = df["uid"].map(lambda u: str(WAV_DIR / split / f"{u}.wav"))
    return df[["uid", "Dialogue_ID", "Utterance_ID", "Speaker", "text", "Emotion", "label", "video", "wav"]]
