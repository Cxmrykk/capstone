"""Training execution and lifecycle management for Text2Cypher LoRA fine-tuning.

Orchestrates dataset tokenization, parameter-efficient model configuration,
checkpoint handling, evaluation scheduling, and final artifact export.

Memory strategy for a 16 GB T4: both target models have 248k-262k token
vocabularies, so a full [batch, seq_len, vocab] logits tensor (plus its fp32
copy inside cross-entropy) dominates VRAM. Prompt tokens are masked out of the
loss anyway, so TargetOnlyLossTrainer asks the model for logits over only the
trailing positions that contain target Cypher tokens.

Exit codes raised here: 3 = non-finite loss, 4 = paused at the wall-clock
limit before the schedule finished (resume with --resume auto).
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import RunConfig
from src.data.dataset import (
    Text2CypherCollator,
    load_raw_split,
    make_train_val_split,
    subsample,
    tokenize_examples,
)
from src.logging_utils import get_logger
from src.training.checkpointing import (
    HubCheckpointSync,
    build_callbacks,
    resolve_resume,
)
from src.training.model_loader import (
    dtype_flags,
    load_model_and_tokenizer,
    select_dtype,
    trainable_parameter_summary,
)

log = get_logger(__name__)

EXIT_NONFINITE_LOSS = 3
EXIT_TIME_LIMIT = 4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _filter_training_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filters argument dictionary against available TrainingArguments fields for cross-version compatibility."""
    from transformers import TrainingArguments

    valid = {f.name for f in dataclasses.fields(TrainingArguments)}
    kept, dropped = {}, []

    if "eval_strategy" in kwargs and "eval_strategy" not in valid:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    for k, v in kwargs.items():
        if k in valid:
            kept[k] = v
        else:
            dropped.append(k)
    if dropped:
        log.debug("Filtered unhandled TrainingArguments: %s", ", ".join(dropped))
    return kept


def _forward_params(model) -> Set[str]:
    """Collects forward() parameter names of the wrapped base model and its wrapper."""
    names: Set[str] = set()
    candidates = []
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass
    candidates.append(model)
    for module in candidates:
        try:
            names |= set(inspect.signature(module.forward).parameters)
        except (TypeError, ValueError):
            continue
    return names


def _logits_to_keep_kwarg(params: Set[str]) -> Optional[str]:
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def _report_upload_failure(sync: Optional[HubCheckpointSync], context: str) -> bool:
    """Logs a prominent error if the most recent Hub upload failed. Returns True if it did."""
    if sync is None or not sync.last_upload_error:
        return False
    log.error("=" * 66)
    log.error("HUB UPLOAD FAILED (%s): %s", context, sync.last_upload_error)
    log.error("Nothing from this session is safely on the Hub. If the Colab runtime is")
    log.error("reset, the next session will NOT resume from this point. Fix the upload")
    log.error("error, then push the local checkpoint manually before the runtime ends:")
    log.error("  python app.py checkpoint push --config <cfg> --path <output_dir>/checkpoint-<step>")
    log.error("=" * 66)
    return True


@dataclass
class TokenTypeIdsCollator:
    """Adds an all-zero (text-only) token_type_ids tensor for models that expect it.

    Gemma multimodal wrappers use token_type_ids to build bidirectional masks over
    image tokens and may require them in training mode. All zeros means "text",
    which leaves the causal mask unchanged.
    """
    base: Text2CypherCollator

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        out = self.base(batch)
        out["token_type_ids"] = torch.zeros_like(out["input_ids"])
        return out


def _build_trainer_class(keep_kwarg: Optional[str]):
    """Returns Trainer, or a subclass that computes loss on target positions only."""
    from transformers import Trainer

    if keep_kwarg is None:
        return Trainer

    class TargetOnlyLossTrainer(Trainer):
        """Requests logits for the trailing target span only and computes CE there.

        Logits at position i predict token i+1, so to score every label from the
        earliest target index `f` in the batch we need logits from position f-1
        onwards: keep = seq_len - f + 1.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # We return a per-micro-batch mean, so let Trainer divide by the
            # gradient accumulation steps itself.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch
            import torch.nn.functional as F

            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

            seq_len = labels.shape[1]
            mask = labels.ne(-100)
            if not bool(mask.any()):
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

            has_target = mask.any(dim=1)
            first = mask.int().argmax(dim=1)
            first = torch.where(has_target, first, torch.full_like(first, seq_len))
            keep = int(seq_len - int(first.min().item()) + 1)
            keep = max(2, min(keep, seq_len))

            forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            forward_inputs[keep_kwarg] = keep
            outputs = model(**forward_inputs)

            logits = outputs.logits
            if logits.shape[1] < keep:
                raise RuntimeError(
                    f"Model returned {logits.shape[1]} logit positions but {keep} were requested. "
                    "Set train.target_only_loss: false for this model."
                )
            logits = logits[:, -keep:, :]

            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, seq_len - keep + 1:].to(shift_logits.device)
            n_tokens = shift_labels.ne(-100).sum().clamp(min=1)

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / n_tokens

            return (loss, outputs) if return_outputs else loss

    return TargetOnlyLossTrainer


# --------------------------------------------------------------------------- #
# Training Entry Point
# --------------------------------------------------------------------------- #
def run_training(cfg: RunConfig, resume: str = "auto") -> Path:
    """Executes the LoRA fine-tuning loop."""
    import torch
    from transformers import TrainingArguments, set_seed

    t_start = time.time()
    output_dir = cfg.resolved_output_dir()
    set_seed(cfg.train.seed)

    log.info("=" * 66)
    log.info("Run Name       : %s", cfg.run_name)
    log.info("Model          : %s (%s)", cfg.model.key, cfg.model.resolved_path())
    log.info("Schema Mode    : %s", cfg.data.schema_mode)
    log.info("Output Dir     : %s", output_dir)
    log.info("Fingerprint    : %s", cfg.fingerprint())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info("Accelerator    : %s (%.1f GB, sm_%d%d)",
                 props.name, props.total_memory / 1024 ** 3, props.major, props.minor)
    log.info("Compute Dtype  : %s", select_dtype())
    log.info("=" * 66)

    # Remote checkpoint synchronization initialization
    sync: Optional[HubCheckpointSync] = None
    if cfg.hub.sync:
        try:
            sync = HubCheckpointSync.from_config(cfg)
            sync.ensure_repo()
        except Exception as exc:
            log.error("Hub checkpoint sync unavailable: %s", exc)
            log.error("Training will continue with local checkpointing only.")
            sync = None

    resume_from = resolve_resume(cfg, resume, sync)

    # Initialize model and tokenizer
    model, tokenizer, meta = load_model_and_tokenizer(cfg, for_training=True)
    log.info("Parameters: %s", trainable_parameter_summary(model))
    forward_params = _forward_params(model)

    # Data loading and tokenization
    raw_train = load_raw_split("train", cfg.data.dataset_dir)
    raw_train = subsample(raw_train, cfg.data.max_train_samples, cfg.data.seed)
    train_records, val_records = make_train_val_split(
        raw_train, cfg.data.val_fraction, cfg.data.max_val_samples, cfg.data.seed
    )

    train_ds = tokenize_examples(train_records, cfg, tokenizer, desc="train")
    eval_ds = tokenize_examples(val_records, cfg, tokenizer, desc="val") if val_records else None

    collator: Any = Text2CypherCollator(pad_token_id=tokenizer.pad_token_id)
    if cfg.model.spec.family == "gemma" and "token_type_ids" in forward_params:
        collator = TokenTypeIdsCollator(collator)
        log.info("Supplying text-only token_type_ids to the Gemma forward pass.")

    # Loss strategy
    keep_kwarg: Optional[str] = None
    if cfg.train.target_only_loss and not meta.get("unsloth"):
        keep_kwarg = _logits_to_keep_kwarg(forward_params)
        if keep_kwarg:
            log.info("Target-only loss enabled (forward kwarg: %s).", keep_kwarg)
        else:
            log.warning("Model forward() has no logits_to_keep argument; "
                        "falling back to full-sequence logits (higher VRAM).")
    elif meta.get("unsloth"):
        log.info("Unsloth active; using its fused cross-entropy loss.")
    trainer_cls = _build_trainer_class(keep_kwarg)

    # Training arguments configuration
    flags = dtype_flags()
    args_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": cfg.run_name,
        "per_device_train_batch_size": cfg.train.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.train.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "learning_rate": cfg.train.learning_rate,
        "num_train_epochs": cfg.train.num_train_epochs,
        "max_steps": cfg.train.max_steps,
        "warmup_ratio": cfg.train.warmup_ratio,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "optim": cfg.train.optim,
        "weight_decay": cfg.train.weight_decay,
        "max_grad_norm": cfg.train.max_grad_norm,
        "logging_steps": cfg.train.logging_steps,
        "save_steps": cfg.train.save_steps,
        "save_strategy": "steps",
        "save_total_limit": cfg.train.save_total_limit,
        "eval_strategy": "steps" if eval_ds is not None else "no",
        "eval_steps": cfg.train.eval_steps,
        # Validation only needs the loss. Without this, Trainer gathers the full
        # [batch, seq, vocab] logits for every validation example and OOMs.
        "prediction_loss_only": True,
        "gradient_checkpointing": cfg.train.gradient_checkpointing and not meta.get("unsloth"),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "group_by_length": cfg.train.group_by_length,
        "dataloader_num_workers": cfg.train.dataloader_num_workers,
        "seed": cfg.train.seed,
        "data_seed": cfg.train.seed,
        "report_to": cfg.train.report_to or "none",
        "logging_first_step": True,
        "save_safetensors": True,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        **flags,
    }
    if cfg.train.optim.startswith("paged_") or "8bit" in cfg.train.optim:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            log.warning("bitsandbytes unavailable; falling back to adamw_torch optimizer.")
            args_kwargs["optim"] = "adamw_torch"

    training_args = TrainingArguments(**_filter_training_args(args_kwargs))

    callbacks = build_callbacks(cfg, sync, cfg.fingerprint())

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
        "callbacks": callbacks,
    }

    trainer_params = inspect.signature(trainer_cls.__init__).parameters
    from transformers import Trainer as _BaseTrainer
    base_params = inspect.signature(_BaseTrainer.__init__).parameters
    if "processing_class" in trainer_params or "processing_class" in base_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trainer_cls(**trainer_kwargs)

    # Persist the configuration alongside model checkpoints
    cfg.save(output_dir / "run_config.json")

    oom_types = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                                  getattr(torch.cuda, "OutOfMemoryError", None)) if t)

    log.info("Initiating training loop (resume_from=%s)", resume_from or "scratch")
    try:
        result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user. Saving emergency checkpoint.")
        trainer.save_model(str(output_dir / "interrupted"))
        if sync is not None:
            sync.wait()
            sync.push(output_dir / "interrupted",
                      step=int(trainer.state.global_step), blocking=True)
            _report_upload_failure(sync, "emergency checkpoint")
        raise
    except oom_types:
        log.error(
            "CUDA out of memory. Options for a 16 GB T4: confirm train.target_only_loss is true "
            "and per_device_train_batch_size is 1; lower data.max_seq_length (e.g. 1536); "
            "use the exact_match schema config; or reduce lora.r. Progress up to the last "
            "checkpoint is preserved -- rerun with --resume auto after changing a non-fingerprinted "
            "setting, or under a new --run-name otherwise."
        )
        if sync is not None:
            sync.wait()
            _report_upload_failure(sync, "last checkpoint before OOM")
        raise

    guard = next((cb for cb in callbacks if getattr(cb, "is_nonfinite_guard", False)), None)
    if guard is not None and guard.tripped:
        log.error(
            "Training stopped because the loss became non-finite (likely fp16 overflow). "
            "The final adapter was NOT saved. Try a lower learning rate, a smaller "
            "max_grad_norm, or rerun with model.upcast_frozen_params: true."
        )
        if sync is not None:
            sync.wait()
            _report_upload_failure(sync, "last checkpoint before non-finite loss")
        raise SystemExit(EXIT_NONFINITE_LOSS)

    # A wall-clock stop (Colab session budget) returns from trainer.train()
    # normally, but the adapter is only partially trained. Writing it as "final"
    # would let predict/merge -- and the notebook's skip-if-final logic -- treat
    # an unfinished run as complete. The checkpoint saved at the stop is enough
    # to resume.
    state = trainer.state
    if state.max_steps and state.global_step < state.max_steps:
        upload_failed = False
        if sync is not None:
            sync.wait()
            upload_failed = _report_upload_failure(sync, "time-limit checkpoint")
        if sync is None:
            where = "locally only"
        elif upload_failed:
            where = "locally only (Hub upload FAILED, see above)"
        else:
            where = "and synced to the Hub"
        log.warning(
            "Training paused at step %d/%d (time limit reached). The checkpoint was saved %s. "
            "Rerun with --resume auto to continue. No final adapter was written.",
            state.global_step, state.max_steps, where,
        )
        raise SystemExit(EXIT_TIME_LIMIT)

    # Final artifact export
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metadata = {
        "run_name": cfg.run_name,
        "model_key": cfg.model.key,
        "model_path": cfg.model.resolved_path(),
        "schema_mode": cfg.data.schema_mode,
        "fingerprint": cfg.fingerprint(),
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "train_runtime_s": round(time.time() - t_start, 1),
        "metrics": getattr(result, "metrics", {}),
        "target_modules": meta.get("target_modules"),
        "unsloth": meta.get("unsloth", False),
        "target_only_loss": keep_kwarg is not None,
        "dtype": str(select_dtype()),
        "trainable_parameters": trainable_parameter_summary(model),
        "n_train": len(train_ds),
        "n_val": len(eval_ds) if eval_ds else 0,
        "config": cfg.to_dict(),
    }
    if torch.cuda.is_available():
        metadata["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        metadata["device"] = torch.cuda.get_device_properties(0).name
    (final_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    if sync is not None:
        sync.wait()
        if not sync.push_final(final_dir, metadata):
            _report_upload_failure(sync, "final adapter")
            log.error("The final adapter exists locally at %s and can be used in this session, "
                      "but a fresh session will retrain unless it is uploaded.", final_dir)

    log.info("=" * 66)
    log.info("Training completed in %.1f min at step %d.",
             (time.time() - t_start) / 60, trainer.state.global_step)
    log.info("Saved final adapter: %s", final_dir)
    log.info("=" * 66)
    return final_dir
