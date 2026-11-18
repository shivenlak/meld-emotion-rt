"""Vision extension: does adding the face help? Measured on MELD.

    python scripts/cache_vision.py      # first (face probabilities for dev/test clips)
    python scripts/evaluate_vision.py

Three-way late fusion on top of the trained text+audio head:
    p = (1 - w) * p_text_audio + w * p_face        (only for clips where a face was found)
w is picked on DEV (gold transcripts) per seed from a grid 0..0.6, then applied
on TEST once, with both gold and ASR transcripts. Seed 0's w is saved to
models/vision_fusion.json for the live pipeline (seed 0 is the deployed head).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_head import predict  # noqa: E402
from src.config import CACHE_DIR, EMOTIONS, MODELS_DIR, RESULTS_DIR, VISION_FUSION_FILE  # noqa: E402
from src.data import load_feats, metrics  # noqa: E402
from src.fusion import load_head  # noqa: E402

GRID = np.round(np.arange(0, 0.61, 0.05), 2)
NEUTRAL = EMOTIONS.index("neutral")


def vision(split):
    z = np.load(CACHE_DIR / f"{split}_vision.npz")
    return z["probs"], z["has_face"], z["uid"]


def ta_probs(model, split, text_source):
    feats, y, uids = load_feats(split, ["text", "audio"], text_source=text_source)
    return predict(model, feats)["fused"].softmax(1).numpy(), y.numpy(), uids


def fuse(p_ta, p_face, has_face, w):
    p = p_ta.copy()
    p[has_face] = (1 - w) * p_ta[has_face] + w * p_face[has_face]
    return p


def main():
    pf_dev, hf_dev, uid_dev = vision("dev")
    pf_test, hf_test, uid_test = vision("test")
    rows = {}

    # Vision alone (no face -> predict neutral, the majority class)
    for split, pf, hf in (("dev", pf_dev, hf_dev), ("test", pf_test, hf_test)):
        _, y, _ = load_feats(split, ["text"])
        pred = np.where(hf, pf.argmax(1), NEUTRAL)
        m = metrics(y.numpy(), pred)
        m["face_found_rate"] = round(float(hf.mean()), 4)
        m_face = metrics(y.numpy()[hf], pf[hf].argmax(1))
        rows[f"{split}: face-only (no face -> neutral)"] = {"seeds": [m]}
        rows[f"{split}: face-only, clips with a face"] = {"seeds": [m_face], "n": int(hf.sum())}

    chosen = {}
    for seed in (0, 1, 2):
        model = load_head(MODELS_DIR / f"fused_s{seed}.pt")
        p_dev, y_dev, u = ta_probs(model, "dev", "gold")
        assert (u == uid_dev).all()
        scores = {float(w): metrics(y_dev, fuse(p_dev, pf_dev, hf_dev, w).argmax(1))["weighted_f1"] for w in GRID}
        w = max(scores, key=scores.get)
        chosen[seed] = {"w_face": w, "dev_weighted_f1_by_w": scores}
        for src in ("gold", "asr"):
            p_t, y_t, u = ta_probs(model, "test", src)
            assert (u == uid_test).all()
            for name, ww in ((f"test: text+audio ({src})", 0.0), (f"test: text+audio+face ({src})", w)):
                rows.setdefault(name, {"seeds": []})["seeds"].append(metrics(y_t, fuse(p_t, pf_test, hf_test, ww).argmax(1)))
        rows.setdefault("dev: text+audio+face (gold)", {"seeds": []})["seeds"].append(
            metrics(y_dev, fuse(p_dev, pf_dev, hf_dev, w).argmax(1)))
        rows.setdefault("dev: text+audio (gold)", {"seeds": []})["seeds"].append(metrics(y_dev, p_dev.argmax(1)))

    lines = ["| set / model | weighted F1 | macro F1 | accuracy | seeds |", "|---|---|---|---|---|"]
    for name, r in rows.items():
        cells = []
        for k in ("weighted_f1", "macro_f1", "accuracy"):
            v = [s[k] for s in r["seeds"]]
            r[k + "_mean"], r[k + "_std"] = float(np.mean(v)), float(np.std(v))
            cells.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {len(r['seeds'])} |")
    lines += ["", f"Face found (YuNet, largest face, 5 frames/clip): dev {hf_dev.mean():.1%}, test {hf_test.mean():.1%}.",
              f"Face weight w chosen on dev per seed: {[chosen[s]['w_face'] for s in (0, 1, 2)]}"]
    pc = rows["test: face-only, clips with a face"]["seeds"][0]["per_class_f1"]
    lines += ["", "Face-only per-class F1 (test, clips with a face): "
              + ", ".join(f"{e} {pc[e]:.2f}" for e in EMOTIONS)]
    table = "\n".join(lines)
    print(table)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results_vision.md").write_text(table + "\n")
    (RESULTS_DIR / "metrics_vision.json").write_text(json.dumps({"rows": rows, "chosen": chosen}, indent=1))
    VISION_FUSION_FILE.write_text(json.dumps({"w_face": chosen[0]["w_face"], "chosen_on": "MELD dev, gold transcripts",
                                              "head": "models/fused.pt (seed 0)"}, indent=1))


if __name__ == "__main__":
    main()
