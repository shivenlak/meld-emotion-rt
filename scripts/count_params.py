"""Count every learned parameter on the local inference path, check the 6B cap.

    python scripts/count_params.py

How each model is counted:
  * PyTorch models (Silero VAD, RoBERTa, WavLM, fusion head): load them and
    sum p.numel() over model.parameters().
  * Whisper runs in CTranslate2 format and Qwen3 is 4-bit packed in MLX, so
    numel() of the stored tensors is not the parameter count. We read the
    ORIGINAL checkpoints' safetensors headers from Hugging Face (only the header
    is downloaded, not the weights). For Qwen we cross-check by unpacking the
    4-bit MLX tensors (each uint32 holds 32/bits weights).
Quantization changes bits per parameter, not the number of parameters, so the
original count is the one that counts against the cap.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

CAP = 6_000_000_000


def torch_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def hf_header_count(repo: str) -> int:
    from huggingface_hub import get_safetensors_metadata
    meta = get_safetensors_metadata(repo)
    return sum(meta.parameter_count.values())


def mlx_unpacked_count(repo: str) -> int:
    """Count logical weights inside a quantized MLX model.

    A quantized layer stores: .weight (uint32, each packing 32/bits weights),
    .scales and .biases (per-group constants for de-quantizing, not learned
    weights of the original model). Everything else is stored as-is."""
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm import load
    model, _ = load(repo)
    cfg = json.loads((Path(snapshot_download(repo, allow_patterns=["config.json"])) / "config.json").read_text())
    bits = cfg["quantization"]["bits"]
    flat = dict(tree_flatten(model.parameters()))
    total = 0
    for name, arr in flat.items():
        prefix = name.rsplit(".", 1)[0]
        quantized_layer = flat.get(prefix + ".weight") is not None and flat[prefix + ".weight"].dtype == mx.uint32
        if quantized_layer and name.endswith((".scales", ".biases")):
            continue
        total += arr.size * (32 // bits) if arr.dtype == mx.uint32 else arr.size
    return total


def main():
    rows = []
    from silero_vad import load_silero_vad
    rows.append(("VAD", "Silero VAD (snakers4/silero-vad)", torch_count(load_silero_vad()), "loaded, numel"))
    rows.append(("ASR", f"Whisper {config.ASR_MODEL} ({config.ASR_HF_REPO})", hf_header_count(config.ASR_HF_REPO),
                 "safetensors header of original"))
    from transformers import AutoModel
    rows.append(("Text encoder", config.TEXT_ENCODER, torch_count(AutoModel.from_pretrained(config.TEXT_ENCODER, add_pooling_layer=False)),
                 "loaded, numel"))
    rows.append(("Audio encoder", config.AUDIO_ENCODER, torch_count(AutoModel.from_pretrained(config.AUDIO_ENCODER)),
                 "loaded, numel"))
    head = config.MODELS_DIR / "fused.pt"
    if head.exists():
        from src.fusion import load_head
        rows.append(("Fusion head", "ours (models/fused.pt)", torch_count(load_head(head)), "loaded, numel"))
    llm = hf_header_count(config.LLM_HF_REPO)
    unpacked = mlx_unpacked_count(config.LLM_MLX)
    rows.append(("LLM", f"{config.LLM_HF_REPO} (run as {config.LLM_MLX})", llm,
                 f"safetensors header of original (conservative: the checkpoint stores the tied "
                 f"embedding twice, as embed_tokens and lm_head); unique weights after unpacking MLX 4-bit = {unpacked:,}"))

    # Vision extension (only loaded with --vision)
    import numpy as np
    import onnx
    from huggingface_hub import hf_hub_download
    from onnx import numpy_helper
    from transformers import AutoModelForImageClassification
    yunet = onnx.load(hf_hub_download(config.FACE_DETECTOR_REPO, config.FACE_DETECTOR_FILE))
    n_yunet = int(sum(np.prod(numpy_helper.to_array(t).shape) for t in yunet.graph.initializer))
    vision_rows = [
        ("Face detector (vision ext.)", f"YuNet ({config.FACE_DETECTOR_REPO})", n_yunet, "ONNX initializers, summed"),
        ("Face expression (vision ext.)", config.FACE_EMOTION_MODEL,
         torch_count(AutoModelForImageClassification.from_pretrained(config.FACE_EMOTION_MODEL)), "loaded, numel"),
    ]

    total = sum(r[2] for r in rows)
    total_v = total + sum(r[2] for r in vision_rows)
    lines = ["| stage | model | parameters | how counted |", "|---|---|---:|---|"]
    lines += [f"| {s} | {m} | {n:,} ({n / 1e6:.1f}M) | {how} |" for s, m, n, how in rows]
    lines.append(f"| **total (core: speech)** | | **{total:,} ({total / 1e9:.3f}B)** | cap 6B: "
                 f"{'OK' if total <= CAP else 'OVER'}, {100 * total / CAP:.1f}% used |")
    lines += [f"| {s} | {m} | {n:,} ({n / 1e6:.2f}M) | {how} |" for s, m, n, how in vision_rows]
    lines.append(f"| **total (with vision)** | | **{total_v:,} ({total_v / 1e9:.3f}B)** | cap 6B: "
                 f"{'OK' if total_v <= CAP else 'OVER'}, {100 * total_v / CAP:.1f}% used |")
    table = "\n".join(lines)
    print(table)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "param_budget.md").write_text(table + "\n")
    (config.RESULTS_DIR / "param_budget.json").write_text(json.dumps(
        {"rows": [{"stage": s, "model": m, "params": n, "how": h} for s, m, n, h in rows + vision_rows],
         "total_core": total, "total_with_vision": total_v, "cap": CAP, "under_cap": total_v <= CAP}, indent=1))


if __name__ == "__main__":
    main()
