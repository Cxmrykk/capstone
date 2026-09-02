"""LoRA adapter merging into unquantized base model weights.

Loads the unquantized base model in full or half precision (fp16/bf16),
applies the trained PEFT adapter weights, merges them in-place, and exports
a standalone Hugging Face model directory ready for inference or GGUF conversion.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from src.config import RunConfig
from src.logging_utils import get_logger
from src.paths import merged_dir

log = get_logger(__name__)

_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


def merge_adapter(cfg: RunConfig, adapter_path: str, out_dir: Optional[str] = None,
                  dtype: str = "float16") -> Path:
    """Merges a LoRA adapter into full-precision base weights."""
    import torch
    from peft import PeftModel

    from src.training.model_loader import _dtype_kwarg, _load_base_model, load_tokenizer

    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter}")

    torch_dtype = getattr(torch, _DTYPES.get(dtype, "float16"))
    out = Path(out_dir) if out_dir else merged_dir() / f"{cfg.run_name}-merged"
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model in %s precision...", dtype)
    original_4bit = cfg.model.load_in_4bit
    cfg.model.load_in_4bit = False
    try:
        model = _load_base_model(cfg, quantize=False, dtype=torch_dtype, for_training=False)
    finally:
        cfg.model.load_in_4bit = original_4bit

    log.info("Attaching LoRA adapter from %s ...", adapter)
    model = PeftModel.from_pretrained(model, str(adapter), torch_dtype=torch_dtype)

    log.info("Merging adapter weights into base model layers...")
    model = model.merge_and_unload()
    model = model.to(torch_dtype)

    log.info("Saving merged model to %s ...", out)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")

    tokenizer = load_tokenizer(cfg)
    tokenizer.save_pretrained(str(out))

    # Copy tokenizer templates and processor configurations
    base_path = Path(cfg.model.resolved_path())
    if base_path.exists():
        for name in ("chat_template.jinja", "preprocessor_config.json",
                     "processor_config.json", "generation_config.json"):
            src = base_path / name
            if src.exists() and not (out / name).exists():
                shutil.copy2(src, out / name)

    metadata = {
        "base_model": cfg.model.resolved_path(),
        "model_key": cfg.model.key,
        "adapter": str(adapter),
        "dtype": dtype,
        "run_name": cfg.run_name,
    }
    (out / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    log.info("Model merge successfully completed: %s", out)
    return out
