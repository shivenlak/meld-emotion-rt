| set / model | weighted F1 | macro F1 | accuracy | seeds |
|---|---|---|---|---|
| dev: face-only (no face -> neutral) | 0.252 | 0.167 | 0.266 | 1 |
| dev: face-only, clips with a face | 0.253 | 0.168 | 0.267 | 1 |
| test: face-only (no face -> neutral) | 0.271 | 0.159 | 0.277 | 1 |
| test: face-only, clips with a face | 0.267 | 0.158 | 0.274 | 1 |
| test: text+audio (gold) | 0.620 ± 0.002 | 0.448 ± 0.001 | 0.621 ± 0.002 | 3 |
| test: text+audio+face (gold) | 0.623 ± 0.005 | 0.453 ± 0.006 | 0.626 ± 0.005 | 3 |
| test: text+audio (asr) | 0.530 ± 0.001 | 0.362 ± 0.002 | 0.548 ± 0.003 | 3 |
| test: text+audio+face (asr) | 0.533 ± 0.002 | 0.365 ± 0.003 | 0.551 ± 0.004 | 3 |
| dev: text+audio+face (gold) | 0.613 ± 0.002 | 0.477 ± 0.010 | 0.617 ± 0.005 | 3 |
| dev: text+audio (gold) | 0.610 ± 0.001 | 0.474 ± 0.005 | 0.613 ± 0.004 | 3 |

Face found (YuNet, largest face, 5 frames/clip): dev 99.1%, test 99.1%.
Face weight w chosen on dev per seed: [0.1, 0.1, 0.2]

Face-only per-class F1 (test, clips with a face): anger 0.22, disgust 0.00, fear 0.00, joy 0.34, neutral 0.35, sadness 0.18, surprise 0.01
