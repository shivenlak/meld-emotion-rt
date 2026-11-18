"""Train an emotion head on cached embeddings. Model selection uses DEV only.

    python scripts/train_head.py --modalities text          --name text
    python scripts/train_head.py --modalities audio         --name audio
    python scripts/train_head.py --modalities text audio    --name fused

Trains one model per seed (default 3 seeds), so results can be reported as
mean ± std instead of one lucky run. Saves models/{name}_s{seed}.pt.
Seed 0 is copied to models/{name}.pt, the model the live pipeline loads.

Class imbalance: MELD is ~47% neutral, while fear/disgust are ~2-3%. Plain
cross-entropy learns to say "neutral" a lot. We weight each class's loss by
(1/frequency)^power: power=0 means no weighting, 1 is full inverse frequency,
0.5 is a softer compromise (chosen on dev, see DECISIONS.md).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_feats, metrics  # noqa: E402
from src.fusion import N_CLASSES, EmotionHead, save_head  # noqa: E402


def class_weights(y: torch.Tensor, power: float) -> torch.Tensor:
    freq = torch.bincount(y, minlength=N_CLASSES).float() / len(y)
    w = (1.0 / freq.clamp(min=1e-6)) ** power
    return w / (w * freq).sum()  # normalise so the average weighted sample counts 1


def predict(model, feats, batch=512):
    model.eval()
    n = next(iter(feats.values())).shape[0]
    outs = []
    with torch.no_grad():
        for i in range(0, n, batch):
            o = model({m: x[i:i + batch] for m, x in feats.items()})
            outs.append({k: v for k, v in o.items()})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def train_one(args, seed, tr, ytr, dv, ydv):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_layers = {m: tr[m].shape[1] for m in args.modalities}
    model = EmotionHead(args.modalities, n_layers, hidden=args.hidden, dropout=args.dropout,
                        modality_dropout=args.modality_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    w = class_weights(ytr, args.class_weight_power)
    best, best_state, bad, history = -1, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            out = model({m: tr[m][idx] for m in args.modalities})
            loss = F.cross_entropy(out["fused"], ytr[idx], weight=w)
            if len(args.modalities) > 1:  # aux losses give per-modality probabilities
                loss = loss + 0.5 * sum(F.cross_entropy(out[m], ytr[idx], weight=w) for m in args.modalities)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pred = predict(model, dv)["fused"].argmax(1).numpy()
        m = metrics(ydv.numpy(), pred)
        history.append({"epoch": epoch, "dev_weighted_f1": m["weighted_f1"], "dev_macro_f1": m["macro_f1"]})
        if m["weighted_f1"] > best:
            best, bad = m["weighted_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            bad += 1
            if bad >= args.patience:  # early stopping on dev
                break
    model.load_state_dict(best_state)
    return model, best, best_epoch, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    args = ap.parse_args()

    tr, ytr, _ = load_feats("train", args.modalities)
    dv, ydv, _ = load_feats("dev", args.modalities)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {"args": vars(args), "runs": []}
    for seed in args.seeds:
        model, best, ep, hist = train_one(args, seed, tr, ytr, dv, ydv)
        cfg = {"modalities": args.modalities, "n_layers": {m: tr[m].shape[1] for m in args.modalities},
               "hidden": args.hidden, "dropout": args.dropout, "seed": seed, "dev_weighted_f1": best,
               "best_epoch": ep, "args": vars(args)}
        save_head(model, MODELS_DIR / f"{args.name}_s{seed}.pt", cfg)
        summary["runs"].append({"seed": seed, "dev_weighted_f1": best, "best_epoch": ep, "history": hist})
        print(f"[{args.name}] seed {seed}: best dev weighted F1 {best:.4f} at epoch {ep}")
    devs = [r["dev_weighted_f1"] for r in summary["runs"]]
    summary["dev_weighted_f1_mean"], summary["dev_weighted_f1_std"] = float(np.mean(devs)), float(np.std(devs))
    print(f"[{args.name}] dev weighted F1 {np.mean(devs):.4f} ± {np.std(devs):.4f}")
    shutil.copy(MODELS_DIR / f"{args.name}_s{args.seeds[0]}.pt", MODELS_DIR / f"{args.name}.pt")
    (RESULTS_DIR / f"train_{args.name}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
