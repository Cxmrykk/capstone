"""Evaluation inference pipeline.

Runs inference across dataset partitions using either Transformers or GGUF backends,
extracts Cypher statements, records latency metrics, and writes JSONL outputs.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import RunConfig
from src.data.dataset import load_raw_split, subsample, write_jsonl
from src.data.prompts import extract_cypher, render_prompt, template_supports_system
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

    prompts: List[str] = []
    schema_texts: List[str] = []
    prompt_tokens: List[int] = []

    for rec in records:
        question = rec.get("question") or ""
        schema_text = apply_schema_mode(
            rec.get("schema") or "", question, cfg.data.schema_mode,
            min_nodes=cfg.data.filter_min_nodes,
            keep_patterns_for_kept_nodes=cfg.data.filter_keep_all_patterns_for_kept_nodes,
            similarity_threshold=cfg.data.similarity_threshold,
            similarity_model=cfg.data.similarity_model,
        )
        prompt = render_prompt(
            tokenizer, question, schema_text,
            supports_system_role=supports_system,
            chat_template_kwargs=cfg.model.spec.chat_template_kwargs,
        )
        prompts.append(prompt)
        schema_texts.append(schema_text)
        try:
            prompt_tokens.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        except Exception:
            prompt_tokens.append(-1)

    t0 = time.time()
    raw_outputs = backend.generate(
        prompts,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        batch_size=cfg.generation.batch_size,
    )
    elapsed = time.time() - t0
    backend.close()

    rows: List[Dict[str, Any]] = []
    for rec, schema_text, n_tok, raw in zip(records, schema_texts, prompt_tokens, raw_outputs):
        rows.append({
            "instance_id": rec.get("instance_id"),
            "question": rec.get("question"),
            "gold_cypher": (rec.get("cypher") or "").strip(),
            "predicted_raw": raw,
            "predicted_cypher": extract_cypher(raw),
            "database_reference": rec.get("database_reference"),
            "data_source": rec.get("data_source"),
            "schema_mode": cfg.data.schema_mode,
            "schema_chars": len(schema_text),
            "prompt_tokens": n_tok,
        })

    out = Path(out_path) if out_path else predictions_dir() / _default_out_name(
        cfg, backend_name, split, tag
    )
    write_jsonl(out, rows)

    meta = {
        "run_name": cfg.run_name,
        "tag": tag,
        "model_key": cfg.model.key,
        "model_path": model_path or cfg.model.resolved_path(),
        "adapter": adapter,
        "gguf": gguf,
        "backend": backend_name,
        "backend_info": backend.info(),
        "split": split,
        "n_instances": len(rows),
        "schema_mode": cfg.data.schema_mode,
        "max_new_tokens": cfg.generation.max_new_tokens,
        "temperature": cfg.generation.temperature,
        "batch_size": cfg.generation.batch_size,
        "total_seconds": round(elapsed, 2),
        "seconds_per_item": round(elapsed / max(1, len(rows)), 3),
        "mean_prompt_tokens": (
            round(sum(t for t in prompt_tokens if t >= 0) /
                  max(1, sum(1 for t in prompt_tokens if t >= 0)), 1)
        ),
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
