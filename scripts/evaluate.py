"""Evaluate trained heads -> results table, metrics JSON, confusion matrices.

    python scripts/evaluate.py --split dev     # use while making decisions
    python scripts/evaluate.py --split test    # run ONCE at the end

Rows: majority-class baseline, text-only, audio-only, fused (gold transcript),
and on test also fused/text using Whisper (ASR) transcripts, i.e. what the live
system actually sees. Every model is reported as mean ± std over its seeds.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_head import predict  # noqa: E402
from src.config import CACHE_DIR, EMOTIONS, MODELS_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_feats, metrics  # noqa: E402
from src.fusion import load_head  # noqa: E402

ROWS = [  # (row name, model name, modalities, text source)
    ("text-only (gold transcript)", "text", ["text"], "gold"),
    ("audio-only", "audio", ["audio"], "gold"),
    ("fused text+audio (gold transcript)", "fused", ["text", "audio"], "gold"),
    ("text-only (ASR transcript)", "text", ["text"], "asr"),
    ("fused text+audio (ASR transcript)", "fused", ["text", "audio"], "asr"),
]


def plot_confusion(cm, title, path):
    cm = np.array(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)  # row-normalised = recall per true class
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(7), EMOTIONS)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def word_error_rate(refs: list[str], hyps: list[str]) -> float:
    """WER = (substitutions + deletions + insertions) / reference words, summed over the corpus."""
    errors = total = 0
    for r, h in zip(refs, hyps):
        r, h = _words(r), _words(h)
        d = list(range(len(h) + 1))  # edit distance, one row at a time
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        errors += d[len(h)]
        total += len(r)
    return errors / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    split = args.split
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"split": split, "rows": {}}

    # Majority-class baseline: always predict the most common TRAIN label.
    _, ytr, _ = load_feats("train", ["text"])
    _, y, _ = load_feats(split, ["text"])
    majority = int(np.bincount(ytr.numpy()).argmax())
    out["rows"]["majority class (always '%s')" % EMOTIONS[majority]] = {
        "seeds": [metrics(y.numpy(), np.full(len(y), majority))]}

    for row, name, mods, src in ROWS:
        if src == "asr" and not (CACHE_DIR / f"{split}_asr_text.npz").exists():
            continue
        feats, y, _ = load_feats(split, mods, text_source=src)
        runs = []
        for ckpt in sorted(MODELS_DIR.glob(f"{name}_s*.pt")):
            model = load_head(ckpt)
            pred = predict(model, feats)["fused"].argmax(1).numpy()
            runs.append(metrics(y.numpy(), pred))
        if runs:
            out["rows"][row] = {"seeds": runs}

    # Late-fusion baseline: p = w * p_text + (1 - w) * p_audio from the two
    # separately trained single-modality models. w is picked on DEV per seed.
    for src in ("gold", "asr"):
        if src == "asr" and not (CACHE_DIR / f"{split}_asr_text.npz").exists():
            continue
        runs, ws = [], []
        for seed_ckpt in sorted(MODELS_DIR.glob("text_s*.pt")):
            seed = seed_ckpt.stem.split("_s")[-1]
            tm, am = load_head(seed_ckpt), load_head(MODELS_DIR / f"audio_s{seed}.pt")

            def fused_probs(sp, text_src):
                ft, yy, _ = load_feats(sp, ["text"], text_source=text_src)
                fa, _, _ = load_feats(sp, ["audio"])
                pt = predict(tm, ft)["fused"].softmax(1).numpy()
                pa = predict(am, fa)["fused"].softmax(1).numpy()
                return pt, pa, yy.numpy()

            pt, pa, yd = fused_probs("dev", "gold")
            grid = np.linspace(0, 1, 21)
            w = max(grid, key=lambda w: metrics(yd, (w * pt + (1 - w) * pa).argmax(1))["weighted_f1"])
            pt, pa, ys = fused_probs(split, src)
            runs.append(metrics(ys, (w * pt + (1 - w) * pa).argmax(1)))
            ws.append(float(w))
        if runs:
            name = f"late fusion: w·text + (1-w)·audio ({'ASR' if src == 'asr' else 'gold'} transcript)"
            out["rows"][name] = {"seeds": runs, "w_text_per_seed": ws}

    lines = [f"| {split} set | weighted F1 | macro F1 | accuracy | seeds |", "|---|---|---|---|---|"]
    for row, r in out["rows"].items():
        for k in ("weighted_f1", "macro_f1", "accuracy"):
            vals = [s[k] for s in r["seeds"]]
            r[k + "_mean"], r[k + "_std"] = float(np.mean(vals)), float(np.std(vals))
        fmt = lambda k: f"{r[k + '_mean']:.3f} ± {r[k + '_std']:.3f}" if len(r["seeds"]) > 1 else f"{r[k + '_mean']:.3f}"
        lines.append(f"| {row} | {fmt('weighted_f1')} | {fmt('macro_f1')} | {fmt('accuracy')} | {len(r['seeds'])} |")
        slug = re.sub(r"[^a-z0-9]+", "_", row.lower()).strip("_")
        plot_confusion(r["seeds"][0]["confusion"], f"{row} [{split}, seed 0]", RESULTS_DIR / f"confusion_{split}_{slug}.png")

    # Per-class F1 table (seed 0) to show which emotions the system can / can't detect.
    lines += ["", "Per-class F1 (seed 0):", "", "| model | " + " | ".join(EMOTIONS) + " |",
              "|---|" + "---|" * len(EMOTIONS)]
    for row, r in out["rows"].items():
        pc = r["seeds"][0]["per_class_f1"]
        lines.append(f"| {row} | " + " | ".join(f"{pc[e]:.2f}" for e in EMOTIONS) + " |")
    asr_json = CACHE_DIR / f"{split}_asr_transcripts.json"
    if asr_json.exists():
        from src.meld import load_split
        hyp = json.loads(asr_json.read_text())
        df = load_split(split)
        out["asr_wer"] = round(word_error_rate(df["text"].tolist(), [hyp[u] for u in df["uid"]]), 4)
        lines += ["", f"ASR word error rate (Whisper {split} transcripts vs MELD gold text): **{out['asr_wer']:.1%}**"]
    table = "\n".join(lines)
    print(table)
    (RESULTS_DIR / f"results_{split}.md").write_text(table + "\n")
    (RESULTS_DIR / f"metrics_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
