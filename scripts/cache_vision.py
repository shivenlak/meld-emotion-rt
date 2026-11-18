"""Face-expression probabilities for every MELD dev/test clip (vision extension).

    python scripts/cache_vision.py

For each clip: 5 evenly spaced frames -> largest face per frame -> ViT FER
probabilities -> mean over frames with a face. No training happens on vision
(off-the-shelf model), so train is not needed: dev picks the fusion weight,
test reports the result.

Output: cache/{split}_vision.npz  probs [N, 7], has_face [N], uid, label
"""
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CACHE_DIR  # noqa: E402
from src.meld import load_split  # noqa: E402
from src.vision import FaceEmotion, video_frames  # noqa: E402


def main():
    fe = FaceEmotion()
    for split in ("dev", "test"):
        df = load_split(split)
        probs, has, t0 = [], [], time.time()
        for path in tqdm(df["video"], desc=split):
            p, _ = fe.predict_frames(video_frames(path, 5)) if Path(path).exists() else (None, 0)
            has.append(p is not None)
            probs.append(p if p is not None else np.full(7, 1 / 7, np.float32))
        np.savez(CACHE_DIR / f"{split}_vision.npz", probs=np.stack(probs).astype(np.float32),
                 has_face=np.array(has), uid=np.array(df["uid"].tolist()), label=np.array(df["label"].tolist()))
        print(f"{split}: {len(df)} clips, face found in {np.mean(has):.1%}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
