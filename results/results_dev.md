| dev set | weighted F1 | macro F1 | accuracy | seeds |
|---|---|---|---|---|
| majority class (always 'neutral') | 0.252 | 0.085 | 0.424 | 1 |
| text-only (gold transcript) | 0.601 ± 0.003 | 0.454 ± 0.012 | 0.608 ± 0.004 | 3 |
| audio-only | 0.480 ± 0.007 | 0.340 ± 0.016 | 0.491 ± 0.016 | 3 |
| fused text+audio (gold transcript) | 0.610 ± 0.001 | 0.474 ± 0.005 | 0.613 ± 0.004 | 3 |
| late fusion: w·text + (1-w)·audio (gold transcript) | 0.612 ± 0.006 | 0.467 ± 0.003 | 0.626 ± 0.010 | 3 |

Per-class F1 (seed 0):

| model | anger | disgust | fear | joy | neutral | sadness | surprise |
|---|---|---|---|---|---|---|---|
| majority class (always 'neutral') | 0.00 | 0.00 | 0.00 | 0.00 | 0.60 | 0.00 | 0.00 |
| text-only (gold transcript) | 0.46 | 0.22 | 0.24 | 0.58 | 0.75 | 0.40 | 0.64 |
| audio-only | 0.48 | 0.00 | 0.28 | 0.34 | 0.64 | 0.41 | 0.35 |
| fused text+audio (gold transcript) | 0.45 | 0.18 | 0.31 | 0.56 | 0.74 | 0.49 | 0.63 |
| late fusion: w·text + (1-w)·audio (gold transcript) | 0.48 | 0.23 | 0.22 | 0.58 | 0.75 | 0.42 | 0.61 |
