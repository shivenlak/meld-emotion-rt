"""Compute frozen RoBERTa / WavLM embeddings once and save them to cache/.

    python scripts/cache_embeddings.py                # gold-text + audio for train/dev/test
    python scripts/cache_embeddings.py --asr-test     # also: Whisper transcripts of test + their text embeddings

Output per split (float16 to halve disk use):
    cache/{split}_text.npz   emb [N, 13, 768], uid, label
    cache/{split}_audio.npz  emb [N, 13, 768], uid, label, has_audio
    cache/test_asr_text.npz  same as test_text but from ASR transcripts
    cache/test_asr_transcripts.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CACHE_DIR  # noqa: E402
from src.meld import SPLITS, load_split  # noqa: E402


def read_wav(path: str):
    if not Path(path).exists():
        return None
    a, _ = sf.read(path, dtype="float32")
    return a if len(a) > 0 else None


def cache_text(enc, texts, out_path, uids, labels):
    embs = []
    for i in tqdm(range(0, len(texts), 64), desc=out_path.name):
        embs.append(enc.encode(texts[i:i + 64]).astype(np.float16))
    np.savez(out_path, emb=np.concatenate(embs), uid=np.array(uids), label=np.array(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-test", action="store_true")
    ap.add_argument("--skip-gold", action="store_true")
    args = ap.parse_args()
    CACHE_DIR.mkdir(exist_ok=True)

    if not args.skip_gold:
        from src.encoders import AudioEncoder, TextEncoder
        tenc, aenc = TextEncoder(), AudioEncoder()
        for split in SPLITS:
            df = load_split(split)
            uids, labels = df["uid"].tolist(), df["label"].tolist()
            cache_text(tenc, [t or " " for t in df["text"]], CACHE_DIR / f"{split}_text.npz", uids, labels)

            embs, has = [], []
            for path in tqdm(df["wav"], desc=f"{split}_audio"):
                a = read_wav(path)
                has.append(a is not None)
                e = aenc.encode(a) if a is not None else np.zeros((13, 768), np.float32)
                embs.append(e.astype(np.float16))
            np.savez(CACHE_DIR / f"{split}_audio.npz", emb=np.stack(embs), uid=np.array(uids),
                     label=np.array(labels), has_audio=np.array(has))
            print(f"{split}: {len(df)} utterances, {len(df) - sum(has)} without audio")

    if args.asr_test:
        from src.asr import ASR
        from src.encoders import TextEncoder
        asr, tenc = ASR(), TextEncoder()
        df = load_split("test")
        transcripts, t0 = {}, time.time()
        for uid, path in tqdm(zip(df["uid"], df["wav"]), total=len(df), desc="asr test"):
            a = read_wav(path)
            transcripts[uid] = asr.transcribe(a) if a is not None else ""
        print(f"ASR on {len(df)} test clips took {time.time() - t0:.0f}s on {asr.device}")
        (CACHE_DIR / "test_asr_transcripts.json").write_text(json.dumps(transcripts, indent=1))
        texts = [transcripts[u] or " " for u in df["uid"]]
        cache_text(tenc, texts, CACHE_DIR / "test_asr_text.npz", df["uid"].tolist(), df["label"].tolist())


if __name__ == "__main__":
    main()
