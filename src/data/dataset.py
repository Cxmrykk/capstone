"""Dataset loading, prompt construction, and tokenization for Text2Cypher training.

Handles Parquet loading, reproducible train/validation splitting, token budget
enforcement, and manual label masking to compute loss exclusively over target Cypher tokens.
"""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import RunConfig
from src.data.prompts import build_messages, render_prompt, template_supports_system
from src.data.schema_filter import apply_schema_mode, trim_schema_text
from src.logging_utils import get_logger
from src.paths import dataset_dir

log = get_logger(__name__)

_SPLIT_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

_DB_FIELDS = ("database_reference_alias", "database_reference", "database")


# --------------------------------------------------------------------------- #
# Raw Loading
# --------------------------------------------------------------------------- #
def _parquet_path(split: str, override: Optional[str] = None) -> Path:
    if split not in _SPLIT_FILES:
        raise ValueError(f"Split must be one of {list(_SPLIT_FILES)}, got {split!r}")
    root = dataset_dir(override)
    path = root / _SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}.\n"
            "Run ./download_data.sh to download the benchmark dataset."
        )
    return path


def load_raw_split(split: str, dataset_dir_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads a dataset partition as a list of dictionaries."""
    path = _parquet_path(split, dataset_dir_override)
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        records = table.to_pylist()
    except ImportError:
        import pandas as pd
        records = pd.read_parquet(path).to_dict(orient="records")

    for i, rec in enumerate(records):
        rec.setdefault("instance_id", f"{split}_{i}")
        for f in _DB_FIELDS:
            if f in rec and rec[f]:
                rec["database_reference"] = rec[f]
                break
        rec.setdefault("database_reference", None)
        rec.setdefault("data_source", "unknown")
    log.info("Loaded %d rows from %s", len(records), path.name)
    return records


def list_database_aliases(dataset_dir_override: Optional[str] = None
                          ) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for split in ("train", "test"):
        try:
            for rec in load_raw_split(split, dataset_dir_override):
                alias = rec.get("database_reference") or "(none)"
                counts[alias] = counts.get(alias, 0) + 1
        except FileNotFoundError as exc:
            log.warning("%s", exc)
    return sorted(counts.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------- #
# Splitting & Sampling
# --------------------------------------------------------------------------- #
def subsample(records: Sequence[Dict[str, Any]], limit: Optional[int],
              seed: int) -> List[Dict[str, Any]]:
    """Deterministically samples records using a fixed random seed."""
    records = list(records)
    if limit is None or limit >= len(records):
        return records
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(records)), limit))
    return [records[i] for i in idx]


def make_train_val_split(records: Sequence[Dict[str, Any]], val_fraction: float,
                         max_val: int, seed: int
                         ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits a training list into training and validation sets while leaving test untouched."""
    records = list(records)
    if val_fraction <= 0:
        return records, []
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    n_val = min(max_val, max(1, int(len(records) * val_fraction)))
    val_idx = set(order[:n_val])
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [records[i] for i in sorted(val_idx)]
    return train, val


# --------------------------------------------------------------------------- #
# Prompt Construction
# --------------------------------------------------------------------------- #
@dataclass
class BuiltExample:
    instance_id: str
    question: str
    schema_text: str
    prompt: str
    target: str
    database_reference: Optional[str]
    data_source: str


def build_example(record: Dict[str, Any], cfg: RunConfig, tokenizer=None,
                  supports_system: Optional[bool] = None) -> BuiltExample:
    question = record.get("question") or ""
    raw_schema = record.get("schema") or ""
    target = (record.get("cypher") or "").strip()

    schema_text = apply_schema_mode(
        raw_schema,
        question,
        cfg.data.schema_mode,
        min_nodes=cfg.data.filter_min_nodes,
        keep_patterns_for_kept_nodes=cfg.data.filter_keep_all_patterns_for_kept_nodes,
        similarity_threshold=cfg.data.similarity_threshold,
        similarity_model=cfg.data.similarity_model,
    )

    if tokenizer is not None:
        prompt = render_prompt(
            tokenizer, question, schema_text,
            supports_system_role=supports_system,
            chat_template_kwargs=cfg.model.spec.chat_template_kwargs,
        )
    else:
        messages = build_messages(question, schema_text, supports_system_role=True)
        prompt = "\n\n".join(m["content"] for m in messages)

    return BuiltExample(
        instance_id=str(record.get("instance_id")),
        question=question,
        schema_text=schema_text,
        prompt=prompt,
        target=target,
        database_reference=record.get("database_reference"),
        data_source=record.get("data_source", "unknown"),
    )


# --------------------------------------------------------------------------- #
# Tokenization & Dataset Construction
# --------------------------------------------------------------------------- #
class Text2CypherDataset:
    """Lightweight PyTorch Dataset holding pre-tokenized inputs and label masks."""

    def __init__(self, features: List[Dict[str, List[int]]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]


def _target_ids(tokenizer, target: str) -> List[int]:
    # Use text= kwarg explicitly to prevent Processors from treating it as an image
    ids = tokenizer(text=target, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None and (not ids or ids[-1] != eos):
        ids = ids + [eos]
    return ids


def tokenize_examples(
    records: Sequence[Dict[str, Any]],
    cfg: RunConfig,
    tokenizer,
    desc: str = "",
) -> Text2CypherDataset:
    """Tokenizes examples and constructs label masks (-100 for prompt tokens)."""
    max_len = cfg.data.max_seq_length
    supports_system = cfg.model.spec.supports_system_role
    if supports_system is None:
        supports_system = template_supports_system(tokenizer)

    features: List[Dict[str, List[int]]] = []
    n_trimmed = 0
    n_dropped = 0

    for record in records:
        question = record.get("question") or ""
        raw_schema = record.get("schema") or ""
        target = (record.get("cypher") or "").strip()
        if not target:
            n_dropped += 1
            continue

        schema_text = apply_schema_mode(
            raw_schema, question, cfg.data.schema_mode,
            min_nodes=cfg.data.filter_min_nodes,
            keep_patterns_for_kept_nodes=cfg.data.filter_keep_all_patterns_for_kept_nodes,
            similarity_threshold=cfg.data.similarity_threshold,
            similarity_model=cfg.data.similarity_model,
        )

        target_ids = _target_ids(tokenizer, target)
        if len(target_ids) + 32 > max_len:
            n_dropped += 1
            continue

        prompt_ids: List[int] = []
        trimmed_this_row = False
        for attempt in range(5):
            prompt = render_prompt(
                tokenizer, question, schema_text,
                supports_system_role=supports_system,
                chat_template_kwargs=cfg.model.spec.chat_template_kwargs,
            )
            # Use text= kwarg explicitly
            prompt_ids = tokenizer(text=prompt, add_special_tokens=False)["input_ids"]
            overflow = len(prompt_ids) + len(target_ids) - max_len
            if overflow <= 0:
                break
            trimmed_this_row = True
            schema_text = trim_schema_text(schema_text, int(overflow * 3.5) + 64)
            if attempt == 4:
                keep = max_len - len(target_ids)
                prompt_ids = prompt_ids[-keep:] if keep > 0 else []

        if trimmed_this_row:
            n_trimmed += 1

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]

        if all(l == -100 for l in labels):
            n_dropped += 1
            continue

        features.append({"input_ids": input_ids, "labels": labels})

    log.info(
        "%sTokenized %d examples (schema_mode=%s, max_len=%d); %d schema-trimmed, %d dropped.",
        f"[{desc}] " if desc else "", len(features), cfg.data.schema_mode,
        max_len, n_trimmed, n_dropped,
    )
    return Text2CypherDataset(features)


# --------------------------------------------------------------------------- #
# Data Collator
# --------------------------------------------------------------------------- #
@dataclass
class Text2CypherCollator:
    """Pads dynamic batches and pads labels with -100 to ignore prompt loss."""

    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, labels, attention = [], [], []
        for f in batch:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [self.label_pad_token_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Data Inspection Helpers
# --------------------------------------------------------------------------- #
def dataset_stats(dataset_dir_override: Optional[str] = None) -> str:
    lines: List[str] = ["Dataset: text2cypher-2024v1", "=" * 60]
    total = 0
    for split in ("train", "test"):
        try:
            records = load_raw_split(split, dataset_dir_override)
        except FileNotFoundError as exc:
            lines.append(f"{split}: MISSING -- {exc}")
            continue
        total += len(records)
        sources: Dict[str, int] = {}
        dbs: Dict[str, int] = {}
        q_len, c_len, s_len = [], [], []
        with_db = 0
        for r in records:
            sources[r.get("data_source", "unknown")] = sources.get(r.get("data_source", "unknown"), 0) + 1
            alias = r.get("database_reference")
            if alias:
                with_db += 1
                dbs[alias] = dbs.get(alias, 0) + 1
            q_len.append(len(r.get("question") or ""))
            c_len.append(len(r.get("cypher") or ""))
            s_len.append(len(r.get("schema") or ""))

        lines.append(f"\n[{split}] {len(records)} instances")
        lines.append(f"  with database access : {with_db} ({with_db / max(1, len(records)):.1%})")
        lines.append(f"  distinct databases   : {len(dbs)}")
        lines.append(f"  question chars       : median {statistics.median(q_len):.0f}, "
                     f"max {max(q_len)}")
        lines.append(f"  cypher chars         : median {statistics.median(c_len):.0f}, "
                     f"max {max(c_len)}")
        lines.append(f"  schema chars         : median {statistics.median(s_len):.0f}, "
                     f"max {max(s_len)}")
        lines.append("  data sources:")
        for name, count in sorted(sources.items(), key=lambda kv: -kv[1])[:12]:
            lines.append(f"    {count:>7}  {name}")

    lines.append("\n" + "=" * 60)
    lines.append(f"Total instances: {total}")
    return "\n".join(lines)


def preview_example(cfg: RunConfig, index: int = 0, split: str = "train",
                    use_tokenizer: bool = True) -> str:
    records = load_raw_split(split, cfg.data.dataset_dir)
    if index >= len(records):
        raise IndexError(f"Index {index} out of range ({len(records)} rows in {split})")
    record = records[index]

    tokenizer = None
    if use_tokenizer:
        try:
            from src.training.model_loader import load_tokenizer
            tokenizer = load_tokenizer(cfg)
        except Exception as exc:
            log.warning("Could not load tokenizer (%s); rendering raw prompt body.", exc)

    example = build_example(record, cfg, tokenizer)

    from src.data.schema_filter import parse_schema
    original = parse_schema(record.get("schema") or "")
    pruned = parse_schema(example.schema_text)

    out = [
        "=" * 72,
        f"instance_id   : {example.instance_id}",
        f"data_source   : {example.data_source}",
        f"database      : {example.database_reference}",
        f"schema_mode   : {cfg.data.schema_mode}",
        f"schema size   : {original.size()} -> {pruned.size()}",
        f"schema chars  : {len(record.get('schema') or '')} -> {len(example.schema_text)}",
        "=" * 72,
        "--- PROMPT ---",
        example.prompt,
        "--- TARGET (loss is computed here only) ---",
        example.target,
        "=" * 72,
    ]
    if tokenizer is not None:
        # Use text= kwarg explicitly
        p_ids = tokenizer(text=example.prompt, add_special_tokens=False)["input_ids"]
        t_ids = _target_ids(tokenizer, example.target)
        out.append(f"prompt tokens : {len(p_ids)}")
        out.append(f"target tokens : {len(t_ids)}")
        out.append(f"total         : {len(p_ids) + len(t_ids)} "
                   f"(budget {cfg.data.max_seq_length})")
    return "\n".join(out)


def token_length_stats(cfg: RunConfig, n_samples: int = 1000) -> str:
    from src.training.model_loader import load_tokenizer

    tokenizer = load_tokenizer(cfg)
    records = subsample(load_raw_split("train", cfg.data.dataset_dir),
                        n_samples, cfg.data.seed)
    supports_system = cfg.model.spec.supports_system_role
    if supports_system is None:
        supports_system = template_supports_system(tokenizer)

    lengths: List[int] = []
    for r in records:
        schema_text = apply_schema_mode(
            r.get("schema") or "", r.get("question") or "", cfg.data.schema_mode,
            min_nodes=cfg.data.filter_min_nodes,
            keep_patterns_for_kept_nodes=cfg.data.filter_keep_all_patterns_for_kept_nodes,
            similarity_threshold=cfg.data.similarity_threshold,
            similarity_model=cfg.data.similarity_model,
        )
        prompt = render_prompt(tokenizer, r.get("question") or "", schema_text,
                               supports_system_role=supports_system,
                               chat_template_kwargs=cfg.model.spec.chat_template_kwargs)
        # Use text= kwarg explicitly
        n_prompt = len(tokenizer(text=prompt, add_special_tokens=False)["input_ids"])
        n_target = len(_target_ids(tokenizer, (r.get("cypher") or "").strip()))
        lengths.append(n_prompt + n_target)

    lengths.sort()

    def pct(p: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))]

    over = sum(1 for x in lengths if x > cfg.data.max_seq_length)
    return "\n".join([
        f"Token length distribution ({len(lengths)} samples, "
        f"schema_mode={cfg.data.schema_mode}, model={cfg.model.key})",
        "-" * 60,
        f"  mean   : {statistics.mean(lengths):.0f}" if lengths else "  mean   : n/a",
        f"  median : {pct(0.50)}",
        f"  p90    : {pct(0.90)}",
        f"  p95    : {pct(0.95)}",
        f"  p99    : {pct(0.99)}",
        f"  max    : {lengths[-1] if lengths else 0}",
        f"  over max_seq_length ({cfg.data.max_seq_length}): "
        f"{over} ({over / max(1, len(lengths)):.1%}) -- these get schema-trimmed",
    ])


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
