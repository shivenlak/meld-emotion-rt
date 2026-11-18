| stage | model | parameters | how counted |
|---|---|---:|---|
| VAD | Silero VAD (snakers4/silero-vad) | 462,594 (0.5M) | loaded, numel |
| ASR | Whisper small.en (openai/whisper-small.en) | 241,734,144 (241.7M) | safetensors header of original |
| Text encoder | FacebookAI/roberta-base | 124,055,040 (124.1M) | loaded, numel |
| Audio encoder | microsoft/wavlm-base-plus | 94,381,936 (94.4M) | loaded, numel |
| Fusion head | ours (models/fused.pt) | 533,551 (0.5M) | loaded, numel |
| LLM | Qwen/Qwen3-1.7B (run as mlx-community/Qwen3-1.7B-4bit) | 2,031,739,904 (2031.7M) | safetensors header of original (conservative: the checkpoint stores the tied embedding twice, as embed_tokens and lm_head); unique weights after unpacking MLX 4-bit = 1,720,574,976 |
| **total (core: speech)** | | **2,492,907,169 (2.493B)** | cap 6B: OK, 41.5% used |
| Face detector (vision ext.) | YuNet (opencv/face_detection_yunet) | 53,121 (0.05M) | ONNX initializers, summed |
| Face expression (vision ext.) | trpakov/vit-face-expression | 85,804,039 (85.80M) | loaded, numel |
| **total (with vision)** | | **2,578,764,329 (2.579B)** | cap 6B: OK, 43.0% used |
