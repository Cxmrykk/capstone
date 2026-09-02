"""Training execution and lifecycle management for Text2Cypher LoRA fine-tuning.

Orchestrates dataset tokenization, parameter-efficient model configuration,
checkpoint handling, evaluation scheduling, and final artifact export.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

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


def run_training(cfg: RunConfig, resume: str = "auto") -> Path:
    """Executes the LoRA fine-tuning loop."""
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

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

    # Data loading and tokenization
    raw_train = load_raw_split("train", cfg.data.dataset_dir)
    raw_train = subsample(raw_train, cfg.data.max_train_samples, cfg.data.seed)
    train_records, val_records = make_train_val_split(
        raw_train, cfg.data.val_fraction, cfg.data.max_val_samples, cfg.data.seed
    )

    train_ds = tokenize_examples(train_records, cfg, tokenizer, desc="train")
    eval_ds = tokenize_examples(val_records, cfg, tokenizer, desc="val") if val_records else None

    collator = Text2CypherCollator(pad_token_id=tokenizer.pad_token_id)

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

    import inspect
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # Persist the configuration alongside model checkpoints
    cfg.save(output_dir / "run_config.json")

    log.info("Initiating training loop (resume_from=%s)", resume_from or "scratch")
    try:
        result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user. Saving emergency checkpoint.")
        trainer.save_model(str(output_dir / "interrupted"))
        if sync is not None:
            sync.push(output_dir / "interrupted",
                      step=int(trainer.state.global_step), blocking=True)
        raise

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
        "dtype": str(select_dtype()),
        "trainable_parameters": trainable_parameter_summary(model),
        "n_train": len(train_ds),
        "n_val": len(eval_ds) if eval_ds else 0,
        "config": cfg.to_dict(),
    }
    (final_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    if sync is not None:
        sync.wait()
        sync.push_final(final_dir, metadata)

    log.info("=" * 66)
    log.info("Training completed in %.1f min at step %d.",
             (time.time() - t_start) / 60, trainer.state.global_step)
    log.info("Saved final adapter: %s", final_dir)
    log.info("=" * 66)
    return final_dir
