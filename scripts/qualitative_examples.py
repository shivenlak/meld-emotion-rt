"""Pick qualitative test examples by RULE (not by hand), then run each through
the real end-to-end pipeline (wav -> ASR -> fusion -> LLM reply).

    python scripts/qualitative_examples.py

Categories, chosen from seed-0 predictions on the ASR-transcript test set:
  tone_overrides_words : text-only head wrong, fused head right, and the audio
                         head agrees with the fused answer
  correct_confident    : fused right with high confidence, non-neutral
  low_confidence       : fused confidence < 0.45, so the reply should be neutral/curious
  failure              : fused confidently wrong
Within each category, take the highest-scoring example (deterministic), with
a 1.0-6.0 s length so the clip is a realistic single turn.
Writes results/qualitative_examples.md and .json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_head import predict  # noqa: E402
from src.config import EMOTIONS, MODELS_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_feats  # noqa: E402
from src.fusion import load_head  # noqa: E402
from src.meld import load_split  # noqa: E402


def probs(name, mods):
    feats, y, uids = load_feats("test", mods, text_source="asr")
    out = predict(load_head(MODELS_DIR / f"{name}.pt"), feats)
    return {k: torch.softmax(v, 1).numpy() for k, v in out.items()}, y.numpy(), uids


def main():
    pf, y, uids = probs("fused", ["text", "audio"])
    pt, _, _ = probs("text", ["text"])
    df = load_split("test").set_index("uid")
    dur = np.array([sf.info(df.loc[u, "wav"]).duration if Path(df.loc[u, "wav"]).exists() else 0 for u in uids])
    ok_len = (dur >= 1.0) & (dur <= 6.0)

    fused, text, audio = pf["fused"], pt["fused"], pf["audio"]
    fp, tp, ap_ = fused.argmax(1), text.argmax(1), audio.argmax(1)
    conf = fused.max(1)
    picks = {}
    m = ok_len & (tp != y) & (fp == y) & (ap_ == fp)
    picks["tone_overrides_words"] = np.where(m)[0][np.argsort(-(fused[m, y[m]] - text[m, y[m]]))][:2]
    m = ok_len & (fp == y) & (y != EMOTIONS.index("neutral"))
    picks["correct_confident"] = np.where(m)[0][np.argsort(-conf[m])][:1]
    m = ok_len & (conf < 0.45)
    picks["low_confidence"] = np.where(m)[0][np.argsort(conf[m])][:1]
    m = ok_len & (fp != y)
    picks["failure"] = np.where(m)[0][np.argsort(-conf[m])][:1]
    counts = {"fused_fixed_text_mistake": int(((tp != y) & (fp == y)).sum()),
              "of_which_audio_head_agreed": int(((tp != y) & (fp == y) & (ap_ == fp)).sum()),
              "fused_broke_text_correct": int(((tp == y) & (fp != y)).sum()), "n_test": int(len(y))}

    from src.pipeline import Pipeline
    pipe = Pipeline(load_llm=True)
    results, md = [], ["# Qualitative examples (MELD test, chosen by rule, see scripts/qualitative_examples.py)", "",
                       f"Over the whole test set (ASR transcripts, seed 0), compared with the text-only head, "
                       f"the fused head **fixed {counts['fused_fixed_text_mistake']}** clips "
                       f"(in {counts['of_which_audio_head_agreed']} of them the audio head alone also had the right "
                       f"answer) and **broke {counts['fused_broke_text_correct']}** clips that text-only got right "
                       f"(out of {counts['n_test']}). The net gain is small: audio is a real but noisy signal.", ""]
    for cat, idxs in picks.items():
        for i in idxs:
            uid = str(uids[i])
            pipe.tracker.reset()
            pipe.history.clear()
            audio_np, _ = sf.read(df.loc[uid, "wav"], dtype="float32")
            state = pipe.process(audio_np)
            reply = "".join(pipe.respond(state)).strip()
            rec = {"category": cat, "uid": uid, "gold_text": df.loc[uid, "text"], "gold_label": EMOTIONS[y[i]],
                   "speaker": df.loc[uid, "Speaker"],
                   "text_only_pred": EMOTIONS[tp[i]], "audio_only_pred": EMOTIONS[ap_[i]], "state": state,
                   "reply": reply}
            results.append(rec)
            md += [f"## {cat}: `test:{uid}`", "",
                   f"- **Gold:** {rec['speaker']}: \"{rec['gold_text']}\" → **{rec['gold_label']}**",
                   f"- **ASR heard:** \"{state['transcript']}\"",
                   f"- **Text-only head:** {rec['text_only_pred']} · **Audio head:** {rec['audio_only_pred']} · "
                   f"**Fused:** {state['emotion']} ({state['confidence']:.2f})",
                   f"- **Lumi:** {reply}",
                   f"- Run it: `python run_file.py test:{uid}`", ""]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "qualitative_examples.json").write_text(json.dumps({"counts": counts, "examples": results}, indent=1))
    (RESULTS_DIR / "qualitative_examples.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
