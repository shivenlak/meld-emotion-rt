#!/usr/bin/env bash
# Small DEV-only sweep for the fused head: class-weight power x modality dropout.
# Writes results/train_sweep_*.json and results/sweep_fused_dev.md. The test set is never touched.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
out=results/sweep_fused_dev.md
echo "| class-weight power | modality dropout | dev weighted F1 (mean ± std, 3 seeds) | dev macro F1 (seed 0) |" > $out
echo "|---|---|---|---|" >> $out
for cw in 0 0.5 1; do
  for md in 0 0.2 0.4; do
    name="sweep_cw${cw}_md${md}"
    $PY scripts/train_head.py --modalities text audio --name "$name" \
        --class-weight-power "$cw" --modality-dropout "$md" > /dev/null 2>&1
    $PY - "$name" "$cw" "$md" >> $out <<'EOF'
import json, sys, torch
sys.path.insert(0, ".")
from scripts.train_head import predict
from src.data import load_feats, metrics
from src.fusion import load_head
name, cw, md = sys.argv[1:]
s = json.load(open(f"results/train_{name}.json"))
f, y, _ = load_feats("dev", ["text", "audio"])
macro = metrics(y.numpy(), predict(load_head(f"models/{name}_s0.pt"), f)["fused"].argmax(1).numpy())["macro_f1"]
print(f"| {cw} | {md} | {s['dev_weighted_f1_mean']:.4f} ± {s['dev_weighted_f1_std']:.4f} | {macro:.4f} |")
EOF
  done
done
cat $out
