"""Live microphone demo (terminal).

    python run_live.py                 # speak; after ~0.5 s of silence it answers
    python run_live.py --simulate f.wav   # replay a file through the SAME streaming path, in real time
    python run_live.py --vision           # + webcam: face expression as a third modality
    python run_live.py --simulate clip.mp4 --vision   # replay a video: its audio AND its frames, in real time

Streaming path: the mic delivers 32 ms chunks -> Silero VAD decides when an
utterance has ended -> the utterance goes through the pipeline -> JSON state
is printed -> the reply streams token by token.

Latency is measured from the moment the VAD declares end of speech. Note that
the VAD itself waits `min_silence_ms` (500 ms) of silence before declaring the
end, a fixed cost reported separately as `vad_wait`.
"""
import src.quiet  # noqa: F401  (must be first: silences library logs; --verbose to keep them)
import argparse
import json
import queue
import sys
import time

import numpy as np

from src.audio_io import load_audio, quiet_noise
from src.config import SAMPLE_RATE
from src.pipeline import Pipeline
from src.vad import CHUNK, StreamingSegmenter

MIN_SILENCE_MS = 500


def chunks_from_mic(q: queue.Queue):
    import sounddevice as sd

    def callback(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK, callback=callback)
    stream.start()
    while True:
        yield q.get()


def chunks_from_file(path: str, realtime: bool = True):
    audio = load_audio(path)
    audio = np.concatenate([audio, quiet_noise(SAMPLE_RATE)])  # trailing silence ends the last turn
    t0 = time.perf_counter()
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        if realtime:  # pace like a real mic: chunk i is available at time i/16000
            delay = t0 + (i + CHUNK) / SAMPLE_RATE - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        yield audio[i:i + CHUNK]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", help="wav/mp4 file to replay as if it were the mic")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print compact JSON")
    ap.add_argument("--vision", action="store_true", help="add face expression (webcam, or the --simulate video)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index for --vision")
    args = ap.parse_args()

    print("Loading models...", file=sys.stderr)
    pipe = Pipeline(load_llm=not args.no_llm, vision=args.vision)
    print(f"Ready ({pipe.load_seconds}s, classifier={pipe.mode}). "
          f"{'Replaying ' + args.simulate if args.simulate else 'Speak now (Ctrl+C to quit).'}", file=sys.stderr)

    watcher = None
    if args.vision:
        from src.vision import FaceWatcher
        watcher = FaceWatcher(pipe.face, source=args.simulate or args.camera)
        time.sleep(1.0)
        if watcher.error:
            print(f"[vision disabled: {watcher.error}]", file=sys.stderr)
            watcher = None

    seg = StreamingSegmenter(min_silence_ms=MIN_SILENCE_MS)
    source = chunks_from_file(args.simulate) if args.simulate else chunks_from_mic(queue.Queue())
    t_start = time.perf_counter()
    try:
        for chunk in source:
            t = time.perf_counter()
            utt = seg.feed(chunk)
            vad_ms = (time.perf_counter() - t) * 1000  # VAD compute for the final chunk
            if utt is None:
                continue
            face_probs, n_faces = None, 0
            if watcher:  # mean face expression over the time the person was speaking
                now = time.perf_counter()
                face_probs, n_faces = watcher.window_probs(now - len(utt) / SAMPLE_RATE, now)
            state = pipe.process(utt, t_end_speech_ms=(time.perf_counter() - t_start) * 1000, vad_ms=vad_ms,
                                 face_probs=face_probs, face_frames=n_faces)
            state["latency_ms"]["vad_wait"] = MIN_SILENCE_MS
            print(json.dumps(state, indent=None if args.quiet else 2), flush=True)
            if not args.no_llm:
                print("Lumi: ", end="", flush=True)
                for piece in pipe.respond(state):
                    print(piece, end="", flush=True)
                print(f"\n[latency_ms: {json.dumps(state['latency_ms'])}]\n", flush=True)
    except KeyboardInterrupt:
        print("\nbye", file=sys.stderr)


if __name__ == "__main__":
    main()
