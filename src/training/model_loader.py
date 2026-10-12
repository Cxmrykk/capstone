"""Model initialization, tokenizer configuration, and LoRA target module discovery.

Handles device-aware precision selection (fp16 on NVIDIA Turing T4, bf16 on
Ampere+), automatic discovery of linear projection layers for LoRA injection,
memory-lean k-bit preparation for 16 GB GPUs, and optional Unsloth acceleration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.config import RunConfig
from src.logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Precision & Compute Dtype Selection
# --------------------------------------------------------------------------- #
def native_bf16_supported() -> bool:
    """True only when the GPU executes bfloat16 natively (compute capability >= 8.0).

    torch.cuda.is_bf16_supported() also returns True when bf16 can merely be
    emulated, which recent PyTorch builds report for Turing (T4, sm_75). Training
    in emulated bf16 is very slow and unsupported by mixed-precision kernels, so
    the capability is checked directly.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 8


def select_dtype():
    """Selects the optimal torch compute dtype based on hardware capabilities."""
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if native_bf16_supported():
        return torch.bfloat16
    return torch.float16


def dtype_flags() -> Dict[str, bool]:
    """Returns TrainingArguments precision flags corresponding to the selected dtype."""
    import torch

    dtype = select_dtype()
    return {
        "fp16": dtype == torch.float16,
        "bf16": dtype == torch.bfloat16,
    }


def _dtype_kwarg(dtype) -> Dict[str, Any]:
    """Returns the from_pretrained dtype keyword for the installed transformers.

    transformers >= 4.56 accepts `dtype` (and v5 deprecates `torch_dtype`). Its
    from_pretrained signature only exposes **kwargs, so the version is checked
    rather than the signature.
    """
    try:
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) >= Version("4.56.0"):
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


# --------------------------------------------------------------------------- #
# Tokenizer Loader
# --------------------------------------------------------------------------- #
def load_tokenizer(cfg: RunConfig, padding_side: str = "right"):
    """Loads and configures the model tokenizer."""
    from transformers import AutoTokenizer

    path = cfg.model.resolved_path()
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=cfg.model.trust_remote_code
        )
    except Exception as exc:
        log.debug("AutoTokenizer failed (%s); attempting AutoProcessor fallback.", exc)
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=cfg.model.trust_remote_code
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"Could not load tokenizer from {path}") from exc

    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        log.info("Tokenizer has no pad token; defaulting to %r.", tokenizer.pad_token)

    tokenizer.padding_side = padding_side
    return tokenizer


# --------------------------------------------------------------------------- #
# LoRA Target Module Resolution
# --------------------------------------------------------------------------- #
def _linear_types() -> Tuple[type, ...]:
    import torch.nn as nn

    types: Tuple[type, ...] = (nn.Linear,)
    try:
        import bitsandbytes as bnb
        types = types + (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except Exception:
        pass
    return types


def _scan_linear_modules(model, exclude_substrings: List[str]) -> List[str]:
    """Returns full paths of eligible linear modules, outside excluded subtrees."""
    linear_types = _linear_types()
    paths: List[str] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        lowered = full_name.lower()
        if any(bad in lowered for bad in exclude_substrings):
            continue
        leaf = full_name.split(".")[-1]
        if leaf in {"lm_head", "score", "classifier"}:
            continue
        paths.append(full_name)
    return paths


def discover_lora_targets(model, exclude_substrings: List[str]) -> List[str]:
    """Discovers recurring linear projection leaf names (e.g. q_proj) for LoRA."""
    counts: Dict[str, int] = {}
    for path in _scan_linear_modules(model, exclude_substrings):
        leaf = path.split(".")[-1]
        counts[leaf] = counts.get(leaf, 0) + 1

    if not counts:
        raise RuntimeError(
            "No eligible linear projection modules found for LoRA. "
            "Inspect the architecture tree and specify lora.target_modules explicitly."
        )

    # Filter out single-instance projection layers; keep recurring transformer projections
    repeated = [n for n, c in counts.items() if c >= 2] or list(counts)
    ordered = sorted(repeated, key=lambda n: (-counts[n], n))
    log.info("Identified %d LoRA target projection layers: %s",
             len(ordered), ", ".join(ordered))
    return ordered


def discover_lora_target_paths(model, exclude_substrings: List[str]) -> List[str]:
    """Returns the exact module paths to adapt.

    PEFT matches leaf names (e.g. "q_proj") anywhere in the model, including
    subtrees excluded above. Gemma 4's vision/audio towers contain q_proj etc.
    as Gemma4ClippableLinear wrappers, which PEFT rejects. Full paths keep the
    adapter on exactly the modules found here.
    """
    leaves = set(discover_lora_targets(model, exclude_substrings))
    paths = [p for p in _scan_linear_modules(model, exclude_substrings)
             if p.split(".")[-1] in leaves]
    log.info("LoRA will adapt %d modules (exact paths; excluded: %s).",
             len(paths), ", ".join(exclude_substrings))
    return paths


def resolve_target_modules(model, cfg: RunConfig, full_paths: bool = True):
    tm = cfg.lora.target_modules
    if isinstance(tm, list):
        return tm
    if tm == "all-linear":
        return "all-linear"
    excludes = cfg.model.spec.lora_exclude_substrings
    if full_paths:
        return discover_lora_target_paths(model, excludes)
    return discover_lora_targets(model, excludes)


def _summarise_targets(targets) -> Any:
    """Compact description of target modules for logs and run metadata."""
    if not isinstance(targets, list):
        return targets
    leaves = sorted({t.split(".")[-1] for t in targets})
    return {"n_modules": len(targets), "leaf_names": leaves}


# --------------------------------------------------------------------------- #
# k-bit Preparation
# --------------------------------------------------------------------------- #
def _prepare_kbit_lean(model, gradient_checkpointing: bool) -> None:
    """Memory-lean replacement for peft.prepare_model_for_kbit_training.

    PEFT's helper upcasts every non-quantized parameter to fp32. On Gemma 4 that
    includes the per-layer embedding tables (billions of parameters), which on its
    own is enough to exhaust a 16 GB T4. Here frozen weights stay in the load
    dtype; only the LoRA weights are trained, in fp32 (see _upcast_trainable_to_fp32).
    """
    for param in model.parameters():
        param.requires_grad_(False)

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception as exc:
            log.debug("enable_input_require_grads failed (%s); continuing.", exc)


def _upcast_trainable_to_fp32(model) -> int:
    """Casts any half-precision trainable parameters to fp32.

    fp16 mixed precision uses a GradScaler, which refuses to unscale fp16
    gradients. Adapters are tiny, so fp32 copies cost almost nothing.
    """
    import torch

    count = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()
            count += 1
    if count:
        log.info("Upcast %d trainable tensors to fp32 for mixed-precision training.", count)
    return count


# --------------------------------------------------------------------------- #
# Base Model Loading
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
            if "attn_implementation" in str(exc) or "sdpa" in str(exc).lower() \
                    or "flash" in str(exc).lower():
                kwargs["attn_implementation"] = "eager"

    raise RuntimeError(
        f"Failed to load model from {path}. Last error: {last_error}\n"
        "Ensure transformers is updated or set model.trust_remote_code: true."
    )


def _try_unsloth(cfg: RunConfig, for_training: bool):
    """Attempts Unsloth acceleration; returns None if unsupported or unavailable."""
    if cfg.model.use_unsloth == "no":
        log.info("Unsloth disabled by config (use_unsloth: no).")
        return None
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        if cfg.model.use_unsloth == "yes":
            raise RuntimeError(f"use_unsloth=yes but unsloth could not be imported: {exc}") from exc
        log.info("Unsloth unavailable (%s); proceeding with standard Transformers + PEFT.", type(exc).__name__)
        return None

    import torch

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.resolved_path(),
            max_seq_length=cfg.data.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.model.load_in_4bit,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        log.info("Loaded model via Unsloth acceleration engine.")
        return model, tokenizer
    except Exception as exc:
        if cfg.model.use_unsloth == "yes":
            raise
        log.warning("Unsloth acceleration failed for %s (%s). Falling back to Transformers + PEFT.",
                    cfg.model.key, exc)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def load_model_and_tokenizer(cfg: RunConfig, for_training: bool = True,
                             adapter_path: Optional[str] = None,
                             model_path_override: Optional[str] = None):
    """Initializes and returns (model, tokenizer, metadata)."""
    if model_path_override:
        cfg.model.path = model_path_override

    meta: Dict[str, Any] = {"backend": "transformers", "unsloth": False}
    dtype = select_dtype()

    unsloth_result = _try_unsloth(cfg, for_training) if for_training and not adapter_path else None

    if unsloth_result is not None:
        model, tokenizer = unsloth_result
        meta["unsloth"] = True
        from unsloth import FastLanguageModel

        # Unsloth expects leaf names; it applies its own module filtering.
        targets = resolve_target_modules(model, cfg, full_paths=False)
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
        meta["target_modules"] = _summarise_targets(targets)
        _finalise(model, tokenizer, cfg, for_training)
        return model, tokenizer, meta

    tokenizer = load_tokenizer(cfg, padding_side="right" if for_training else "left")
    model = _load_base_model(cfg, quantize=cfg.model.load_in_4bit, dtype=dtype,
                             for_training=for_training)

    if adapter_path:
        from peft import PeftModel

        log.info("Loading LoRA adapter weights from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=for_training)
        meta["adapter"] = adapter_path
        if for_training:
            _upcast_trainable_to_fp32(model)
        _finalise(model, tokenizer, cfg, for_training)
        return model, tokenizer, meta

    if for_training:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.model.load_in_4bit:
            if cfg.model.upcast_frozen_params:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=cfg.train.gradient_checkpointing,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            else:
                _prepare_kbit_lean(model, cfg.train.gradient_checkpointing)
        elif cfg.train.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        targets = resolve_target_modules(model, cfg, full_paths=True)
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
        _upcast_trainable_to_fp32(model)
        meta["target_modules"] = _summarise_targets(targets)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    _finalise(model, tokenizer, cfg, for_training)
    return model, tokenizer, meta


def _finalise(model, tokenizer, cfg: RunConfig, for_training: bool) -> None:
    try:
        model.config.use_cache = not for_training
    except Exception:
        pass

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
