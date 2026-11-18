"""Run the full pipeline on a file: one input travelling to both outputs.

    python run_file.py path/to/clip.wav          # any audio/video file; VAD splits it into turns
    python run_file.py test:dia3_utt2            # a MELD clip by split:id (one utterance, no VAD split)
    python run_file.py clip.wav --no-llm         # emotion JSON only
    python run_file.py test:dia3_utt2 --vision   # + face expression from the clip's video frames

Prints one JSON state per utterance, then streams the robot's reply.
"""
import src.quiet  # noqa: F401  (must be first: silences library logs; --verbose to keep them)
import argparse
import json
import sys
import time

from src.audio_io import load_audio
from src.config import MELD_DIR, WAV_DIR
from src.pipeline import Pipeline
from src.vad import split_file

MELD_FOLDERS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}


def resolve(arg: str):
    """'test:dia3_utt2' -> path of the cached wav (or original mp4)."""
    if ":" in arg and not arg.startswith("/"):
        split, uid = arg.split(":", 1)
        wav = WAV_DIR / split / f"{uid}.wav"
        if wav.exists():
            return wav, True
        return MELD_DIR / MELD_FOLDERS[split] / f"{uid}.mp4", True
    return arg, False


def resolve_video(arg: str):
    """Video file for --vision: the MELD mp4 for split:id inputs, else the input itself."""
    if ":" in arg and not arg.startswith("/"):
        split, uid = arg.split(":", 1)
        return MELD_DIR / MELD_FOLDERS[split] / f"{uid}.mp4"
    return arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-split", action="store_true", help="treat the whole file as one utterance")
    ap.add_argument("--vision", action="store_true", help="add face expression from the video frames")
    args = ap.parse_args()

    path, is_meld = resolve(args.input)
    pipe = Pipeline(load_llm=not args.no_llm, vision=args.vision)
    print(f"[loaded models in {pipe.load_seconds}s, classifier={pipe.mode}]", file=sys.stderr)

    audio = load_audio(path)
    if is_meld or args.no_split:
        segments = [{"start_ms": 0, "end_ms": len(audio) / 16, "audio": audio}]
        vad_ms = None
    else:
        t = time.perf_counter()
        segments = split_file(audio)
        vad_ms = (time.perf_counter() - t) * 1000  # whole-file VAD time (offline mode)

    video = resolve_video(args.input) if args.vision else None
    for seg in segments:
        face_probs, n_faces = None, 0
        if video:
            from src.vision import video_frames
            frames = video_frames(video, 5, seg["start_ms"] / 1000, seg["end_ms"] / 1000)
            if frames:
                face_probs, n_faces = pipe.face.predict_frames(frames)
            else:
                print("[vision: no video frames in this input]", file=sys.stderr)
        state = pipe.process(seg["audio"], t_end_speech_ms=seg["end_ms"], vad_ms=vad_ms,
                             face_probs=face_probs, face_frames=n_faces)
        print(json.dumps(state, indent=2))
        if not args.no_llm:
            print("Lumi: ", end="", flush=True)
            for piece in pipe.respond(state):
                print(piece, end="", flush=True)
            print(f"\n[latency_ms: {json.dumps(state['latency_ms'])}]\n")


if __name__ == "__main__":
    main()
