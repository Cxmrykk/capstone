"""Evaluation inference pipeline.

Runs inference across dataset partitions using either Transformers or GGUF backends,
extracts Cypher statements, records latency metrics, and writes JSONL outputs.
Prompts are fitted to data.max_seq_length with the same schema-trimming rule
as training (history/030.md).
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import RunConfig
from src.data.dataset import fit_prompt_to_budget, load_raw_split, subsample, write_jsonl
from src.data.prompts import extract_cypher, template_supports_system
from src.data.schema_filter import apply_schema_mode
from src.inference import get_backend
from src.logging_utils import get_logger
from src.paths import predictions_dir

log = get_logger(__name__)


def _default_out_name(cfg: RunConfig, backend_name: str, split: str,
                      tag: Optional[str]) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = tag or cfg.run_name
    return f"{label}__{cfg.model.key}__{cfg.data.schema_mode}__{backend_name}__{split}__{stamp}.jsonl"


def prompt_reserve_tokens(cfg: RunConfig) -> int:
    """Tokens kept free for the generated query when fitting the prompt."""
    reserve = cfg.generation.prompt_target_reserve
    return int(reserve) if reserve is not None else int(cfg.generation.max_new_tokens)


def run_prediction(
    cfg: RunConfig,
    backend_name: str = "hf",
    adapter: Optional[str] = None,
    model_path: Optional[str] = None,
    gguf: Optional[str] = None,
    server_url: Optional[str] = None,
    split: str = "test",
    limit: Optional[int] = None,
    out_path: Optional[str] = None,
    tag: Optional[str] = None,
) -> Path:
    """Runs generation across an evaluation split and outputs predictions with metadata."""
    if adapter and not Path(adapter).exists():
        raise FileNotFoundError(
            f"Adapter not found: {adapter}\n"
            "If training finished in an earlier session, download it first:\n"
            f"  python app.py checkpoint pull --config <cfg> --step final"
        )

    records = load_raw_split(split, cfg.data.dataset_dir)
    limit = limit if limit is not None else cfg.data.max_eval_samples
    records = subsample(records, limit, cfg.data.seed)
    log.info("Running prediction on %d examples from split '%s'.", len(records), split)

    if backend_name == "hf":
        backend = get_backend("hf", cfg=cfg, adapter=adapter, model_path=model_path)
    else:
        backend = get_backend(
            "llamacpp",
            cfg=cfg,
            gguf_path=gguf,
            server_url=server_url,
            tokenizer_path=model_path,
        )

    backend.load()
    tokenizer = backend.tokenizer()
    if tokenizer is None:
        raise RuntimeError("Backend did not provide a tokenizer; cannot construct prompts.")

    supports_system = cfg.model.spec.supports_system_role
    if supports_system is None:
        supports_system = template_supports_system(tokenizer)

    reserve = prompt_reserve_tokens(cfg)
    prompts: List[str] = []
    fitted_rows = []

    for rec in records:
        question = rec.get("question") or ""
        schema_text = apply_schema_mode(
            rec.get("schema") or "", question, cfg.data.schema_mode,
            min_nodes=cfg.data.filter_min_nodes,
            keep_patterns_for_kept_nodes=cfg.data.filter_keep_all_patterns_for_kept_nodes,
            similarity_threshold=cfg.data.similarity_threshold,
            similarity_model=cfg.data.similarity_model,
        )
        fitted = fit_prompt_to_budget(
            tokenizer, question, schema_text, cfg, supports_system, reserve_tokens=reserve,
        )
        prompts.append(fitted.prompt)
        fitted_rows.append(fitted)

    n_trimmed = sum(1 for f in fitted_rows if f.schema_trimmed)
    n_over = sum(1 for f in fitted_rows if f.over_budget)
    log.info("Prompt budget: %d tokens (max_seq_length %d - reserve %d). Schema-trimmed: %d/%d; "
             "still over budget after trimming: %d.",
             cfg.data.max_seq_length - reserve, cfg.data.max_seq_length, reserve,
             n_trimmed, len(fitted_rows), n_over)
    if n_over:
        log.warning("%d prompts exceed the budget even after trimming; they are sent untruncated "
                    "and flagged prompt_over_budget=True.", n_over)

    t0 = time.time()
    raw_outputs = backend.generate(
        prompts,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        batch_size=cfg.generation.batch_size,
    )
    elapsed = time.time() - t0
    backend_info = backend.info()
    backend.close()

    rows: List[Dict[str, Any]] = []
    for rec, fitted, raw in zip(records, fitted_rows, raw_outputs):
        rows.append({
            "instance_id": rec.get("instance_id"),
            "question": rec.get("question"),
            "gold_cypher": (rec.get("cypher") or "").strip(),
            "predicted_raw": raw,
            "predicted_cypher": extract_cypher(raw),
            "database_reference": rec.get("database_reference"),
            "data_source": rec.get("data_source"),
            "schema_mode": cfg.data.schema_mode,
            "schema_chars": len(fitted.schema_text),
            "schema_chars_untrimmed": fitted.schema_chars_untrimmed,
            "schema_trimmed": fitted.schema_trimmed,
            "prompt_over_budget": fitted.over_budget,
            "prompt_tokens": len(fitted.prompt_ids),
        })

    out = Path(out_path) if out_path else predictions_dir() / _default_out_name(
        cfg, backend_name, split, tag
    )
    write_jsonl(out, rows)

    token_counts = [len(f.prompt_ids) for f in fitted_rows]
    meta = {
        "run_name": cfg.run_name,
        "tag": tag,
        "model_key": cfg.model.key,
        "model_path": model_path or cfg.model.resolved_path(),
        "adapter": adapter,
        "gguf": gguf,
        "backend": backend_name,
        "backend_info": backend_info,
        "split": split,
        "n_instances": len(rows),
        "schema_mode": cfg.data.schema_mode,
        "max_seq_length": cfg.data.max_seq_length,
        "prompt_reserve_tokens": reserve,
        "n_schema_trimmed": n_trimmed,
        "n_prompt_over_budget": n_over,
        "max_new_tokens": cfg.generation.max_new_tokens,
        "temperature": cfg.generation.temperature,
        "batch_size": cfg.generation.batch_size,
        "total_seconds": round(elapsed, 2),
        "seconds_per_item": round(elapsed / max(1, len(rows)), 3),
        "mean_prompt_tokens": round(sum(token_counts) / max(1, len(token_counts)), 1),
        "host": platform.node(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    log.info("Saved %d predictions to %s (%.1fs, %.3f s/item).",
             len(rows), out, elapsed, meta["seconds_per_item"])
    log.info("Average prompt token count: %s (schema_mode=%s)",
             meta["mean_prompt_tokens"], cfg.data.schema_mode)
    return out
