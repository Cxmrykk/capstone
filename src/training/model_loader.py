"""Model and tokenizer loading.

Design notes:

* Unsloth is tried first for acceleration but is not required. New architectures
  frequently lag Unsloth support, and a fallback to Transformers+PEFT ensures
  resilience on Colab sessions.
* dtype is chosen from the device, never hardcoded. A T4 reports
  bf16_supported == False and we fall back to fp16 automatically.
* LoRA target modules are discovered by walking the module tree rather than
  hardcoding names, because Qwen3.5's linear-attention blocks and Gemma 4's
  per-layer-embedding blocks do not use the familiar q_proj/k_proj naming
  throughout. Vision/audio towers are excluded.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List, Optional, Tuple

from src.config import RunConfig
from src.logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# dtype selection
# --------------------------------------------------------------------------- #
def select_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            return torch.bfloat16
    return torch.float16


def dtype_flags() -> Dict[str, bool]:
    """TrainingArguments flags matching the selected dtype."""
    import torch

    dtype = select_dtype()
    return {
        "fp16": dtype == torch.float16,
        "bf16": dtype == torch.bfloat16,
    }


def _dtype_kwarg(dtype) -> Dict[str, Any]:
    """`torch_dtype` was renamed to `dtype` in recent transformers."""
    try:
        from transformers import AutoModelForCausalLM
        params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        if "dtype" in params and "torch_dtype" not in params:
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer(cfg: RunConfig, padding_side: str = "right"):
    from transformers import AutoTokenizer

    path = cfg.model.resolved_path()
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=cfg.model.trust_remote_code
        )
    except Exception as exc:
        log.debug("AutoTokenizer failed (%s); trying AutoProcessor.", exc)
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=cfg.model.trust_remote_code
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"Could not obtain a tokenizer from {path}") from exc

    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        log.info("Tokenizer had no pad token; using %r.", tokenizer.pad_token)

    tokenizer.padding_side = padding_side
    return tokenizer


# --------------------------------------------------------------------------- #
# LoRA target discovery
# --------------------------------------------------------------------------- #
def discover_lora_targets(model, exclude_substrings: List[str]) -> List[str]:
    """Collect leaf names of all linear layers eligible for adaptation."""
    import torch.nn as nn

    linear_types: Tuple[type, ...] = (nn.Linear,)
    try:
        import bitsandbytes as bnb
        linear_types = linear_types + (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except Exception:
        pass

    names: Dict[str, int] = {}
    for full_name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        lowered = full_name.lower()
        if any(bad in lowered for bad in exclude_substrings):
            continue
        leaf = full_name.split(".")[-1]
        if leaf in {"lm_head", "score", "classifier"}:
            continue
        names[leaf] = names.get(leaf, 0) + 1

    if not names:
        raise RuntimeError(
            "No eligible linear layers found for LoRA. Inspect the module tree "
            "and set lora.target_modules explicitly in the config."
        )

    # Drop one-off layers (projectors, adapters) that appear only once or twice;
    # we want the repeated transformer-block projections.
    repeated = [n for n, c in names.items() if c >= 2] or list(names)
    ordered = sorted(repeated, key=lambda n: (-names[n], n))
    log.info("Discovered %d LoRA target module names: %s",
             len(ordered), ", ".join(ordered))
    return ordered


def resolve_target_modules(model, cfg: RunConfig):
    tm = cfg.lora.target_modules
    if isinstance(tm, list):
        return tm
    if tm == "all-linear":
        return "all-linear"
    return discover_lora_targets(model, cfg.model.spec.lora_exclude_substrings)


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
def _quant_config(cfg: RunConfig):
    import torch
    from transformers import BitsAndBytesConfig

    if not cfg.model.load_in_4bit:
        return None
    compute_dtype = select_dtype()
    if compute_dtype == torch.float32:
        compute_dtype = torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


_MODEL_CLASS_CASCADE = [
    ("transformers", "AutoModelForCausalLM"),
    ("transformers", "AutoModelForImageTextToText"),
    ("transformers", "AutoModelForVision2Seq"),
    ("transformers", "AutoModel"),
]


def _load_base_model(cfg: RunConfig, quantize: bool, dtype, for_training: bool):
    import importlib

    path = cfg.model.resolved_path()
    kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "attn_implementation": cfg.model.resolved_attn(),
        "low_cpu_mem_usage": True,
    }
    kwargs.update(_dtype_kwarg(dtype))

    if quantize:
        qc = _quant_config(cfg)
        if qc is not None:
            kwargs["quantization_config"] = qc
            kwargs["device_map"] = {"": 0}
    else:
        import torch
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    last_error: Optional[Exception] = None
    for module_name, class_name in _MODEL_CLASS_CASCADE:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            log.info("Loading %s with %s (attn=%s, 4bit=%s, dtype=%s)",
                     path, class_name, kwargs["attn_implementation"], quantize, dtype)
            model = cls.from_pretrained(path, **kwargs)
            return model
        except Exception as exc:
            last_error = exc
            log.debug("%s failed for %s: %s", class_name, path, exc)
            # An unsupported attn implementation is a common, recoverable cause.
            if "attn_implementation" in str(exc) or "flash" in str(exc).lower():
                kwargs["attn_implementation"] = "eager"

    raise RuntimeError(
        f"Could not load model from {path}. Last error: {last_error}\n"
        "If this is a very new architecture, upgrade transformers "
        "(pip install -U transformers) or set model.trust_remote_code: true."
    )


def _try_unsloth(cfg: RunConfig, for_training: bool):
    """Attempt the Unsloth fast path; return None if unavailable/unsupported."""
    if cfg.model.use_unsloth == "no":
        return None
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        if cfg.model.use_unsloth == "yes":
            raise RuntimeError(f"use_unsloth=yes but unsloth is unimportable: {exc}") from exc
        log.info("Unsloth not available (%s); using transformers + peft.", type(exc).__name__)
        return None

    import torch

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.resolved_path(),
            max_seq_length=cfg.data.max_seq_length,
            dtype=None,                       # Unsloth auto-detects; picks fp16 on T4
            load_in_4bit=cfg.model.load_in_4bit,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        log.info("Loaded via Unsloth.")
        return model, tokenizer
    except Exception as exc:
        if cfg.model.use_unsloth == "yes":
            raise
        log.warning("Unsloth could not load %s (%s). Falling back to transformers + peft.",
                    cfg.model.key, exc)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def load_model_and_tokenizer(cfg: RunConfig, for_training: bool = True,
                             adapter_path: Optional[str] = None,
                             model_path_override: Optional[str] = None):
    """Return (model, tokenizer, meta)."""
    import torch

    if model_path_override:
        cfg.model.path = model_path_override

    meta: Dict[str, Any] = {"backend": "transformers", "unsloth": False}
    dtype = select_dtype()

    unsloth_result = _try_unsloth(cfg, for_training) if for_training and not adapter_path else None

    if unsloth_result is not None:
        model, tokenizer = unsloth_result
        meta["unsloth"] = True
        from unsloth import FastLanguageModel

        targets = resolve_target_modules(model, cfg)
        if targets == "all-linear":
            targets = discover_lora_targets(model, cfg.model.spec.lora_exclude_substrings)
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            target_modules=targets,
            use_gradient_checkpointing="unsloth" if cfg.train.gradient_checkpointing else False,
            random_state=cfg.train.seed,
        )
        meta["target_modules"] = targets
        _finalise(model, tokenizer, cfg, for_training)
        return model, tokenizer, meta

    tokenizer = load_tokenizer(cfg, padding_side="right" if for_training else "left")
    model = _load_base_model(cfg, quantize=cfg.model.load_in_4bit, dtype=dtype,
                             for_training=for_training)

    if adapter_path:
        from peft import PeftModel

        log.info("Attaching LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=for_training)
        meta["adapter"] = adapter_path
        _finalise(model, tokenizer, cfg, for_training)
        return model, tokenizer, meta

    if for_training:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.model.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.train.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif cfg.train.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        targets = resolve_target_modules(model, cfg)
        peft_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            task_type="CAUSAL_LM",
            target_modules=targets,
            modules_to_save=cfg.lora.modules_to_save,
        )
        model = get_peft_model(model, peft_cfg)
        meta["target_modules"] = targets
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    _finalise(model, tokenizer, cfg, for_training)
    return model, tokenizer, meta


def _finalise(model, tokenizer, cfg: RunConfig, for_training: bool) -> None:
    """Common post-load fixes."""
    try:
        if for_training:
            model.config.use_cache = False
        else:
            model.config.use_cache = True
    except Exception:
        pass

    # Keep generation config consistent with the tokenizer.
    try:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            if gen_cfg.pad_token_id is None:
                gen_cfg.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    tokenizer.padding_side = "right" if for_training else "left"


def trainable_parameter_summary(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    return f"trainable {trainable:,} / total {total:,} ({pct:.4f}%)"
