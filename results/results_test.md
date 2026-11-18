| test set | weighted F1 | macro F1 | accuracy | seeds |
|---|---|---|---|---|
| majority class (always 'neutral') | 0.313 | 0.093 | 0.481 | 1 |
| text-only (gold transcript) | 0.620 ± 0.005 | 0.453 ± 0.003 | 0.623 ± 0.008 | 3 |
| audio-only | 0.481 ± 0.007 | 0.308 ± 0.019 | 0.495 ± 0.015 | 3 |
| fused text+audio (gold transcript) | 0.620 ± 0.002 | 0.448 ± 0.001 | 0.621 ± 0.002 | 3 |
| text-only (ASR transcript) | 0.515 ± 0.006 | 0.349 ± 0.008 | 0.536 ± 0.010 | 3 |
| fused text+audio (ASR transcript) | 0.530 ± 0.001 | 0.362 ± 0.002 | 0.548 ± 0.003 | 3 |
| late fusion: w·text + (1-w)·audio (gold transcript) | 0.624 ± 0.009 | 0.452 ± 0.010 | 0.636 ± 0.012 | 3 |
| late fusion: w·text + (1-w)·audio (ASR transcript) | 0.539 ± 0.001 | 0.366 ± 0.003 | 0.566 ± 0.005 | 3 |

Per-class F1 (seed 0):

| model | anger | disgust | fear | joy | neutral | sadness | surprise |
|---|---|---|---|---|---|---|---|
| majority class (always 'neutral') | 0.00 | 0.00 | 0.00 | 0.00 | 0.65 | 0.00 | 0.00 |
| text-only (gold transcript) | 0.47 | 0.25 | 0.27 | 0.58 | 0.78 | 0.33 | 0.53 |
| audio-only | 0.40 | 0.12 | 0.10 | 0.36 | 0.66 | 0.31 | 0.32 |
| fused text+audio (gold transcript) | 0.46 | 0.18 | 0.22 | 0.57 | 0.77 | 0.38 | 0.55 |
| text-only (ASR transcript) | 0.35 | 0.19 | 0.09 | 0.40 | 0.70 | 0.33 | 0.43 |
| fused text+audio (ASR transcript) | 0.35 | 0.15 | 0.12 | 0.44 | 0.69 | 0.37 | 0.44 |
| late fusion: w·text + (1-w)·audio (gold transcript) | 0.50 | 0.20 | 0.27 | 0.58 | 0.79 | 0.35 | 0.53 |
| late fusion: w·text + (1-w)·audio (ASR transcript) | 0.38 | 0.20 | 0.08 | 0.41 | 0.71 | 0.34 | 0.45 |

ASR word error rate (Whisper test transcripts vs MELD gold text): **35.0%**
