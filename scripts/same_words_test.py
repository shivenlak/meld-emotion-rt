""""Same words, different tone": a controlled measurement on real MELD test data.

    python scripts/same_words_test.py

Take every test utterance whose exact gold text appears at least twice
in the test set WITH at least two different gold emotions, e.g. "Oh my God!"
labelled surprise in one scene and joy or fear in another. On these, the
text-only head sees identical input and MUST give the same answer every time,
so any accuracy above that ceiling has to come from the audio (tone).

Uses gold transcripts on purpose: it isolates tone from ASR noise.
Reports text-only vs fused accuracy, mean ± std over the 3 seeds.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_head import predict  # noqa: E402
from src.config import EMOTIONS, MODELS_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_feats  # noqa: E402
from src.fusion import load_head  # noqa: E402
from src.meld import load_split  # noqa: E402


def norm(s: str) -> str:
    # EXACT text (only whitespace normalised). Punctuation matters to the text
    # model ("Hey!" vs "Hey."), so stripping it would break the "identical input" premise.
    return " ".join(s.split())


def main():
    df = load_split("test")
    groups = defaultdict(list)
    for i, t in enumerate(df["text"]):
        groups[norm(t)].append(i)
    idx = [i for g, ix in groups.items() if g and len(ix) >= 2 and len(set(df["label"].iloc[ix])) >= 2 for i in ix]
    idx = np.array(sorted(idx))
    y = df["label"].values[idx]

    ft, _, _ = load_feats("test", ["text"])
    ff, _, _ = load_feats("test", ["text", "audio"])
    res = {"text-only": [], "fused": []}
    for seed in (0, 1, 2):
        pt = predict(load_head(MODELS_DIR / f"text_s{seed}.pt"), ft)["fused"].argmax(1).numpy()[idx]
        pf = predict(load_head(MODELS_DIR / f"fused_s{seed}.pt"), ff)["fused"].argmax(1).numpy()[idx]
        res["text-only"].append(float((pt == y).mean()))
        res["fused"].append(float((pf == y).mean()))

    # Best any text-only model could do: for each identical text, predict its most common label.
    ceiling = sum(np.bincount(df["label"].values[ix]).max() for g, ix in groups.items()
                  if g and len(ix) >= 2 and len(set(df["label"].iloc[ix])) >= 2) / len(idx)
    top = sorted(((g, ix) for g, ix in groups.items() if g and len(ix) >= 2 and len(set(df["label"].iloc[ix])) >= 2),
                 key=lambda x: -len(x[1]))[:6]
    examples = [{"text": g, "n": len(ix), "labels": {EMOTIONS[k]: int(v) for k, v in
                                                     enumerate(np.bincount(df["label"].values[ix], minlength=7)) if v}}
                for g, ix in top]
    out = {"n_utterances": int(len(idx)), "n_distinct_texts": len({norm(df['text'].iloc[i]) for i in idx}),
           "text_only_ceiling_accuracy": round(float(ceiling), 4),
           **{f"{k}_accuracy_mean": round(float(np.mean(v)), 4) for k, v in res.items()},
           **{f"{k}_accuracy_std": round(float(np.std(v)), 4) for k, v in res.items()},
           "examples": examples}
    print(json.dumps(out, indent=1))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "same_words_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
