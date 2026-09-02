"""Scoring orchestration for translation metrics, syntax validation, and execution accuracy."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import RunConfig
from src.data.dataset import read_jsonl, write_jsonl
from src.evaluation.metrics import COMPLEXITY_ORDER, score_row, summarise
from src.evaluation.neo4j_client import Neo4jExecutor, resolve_target
from src.logging_utils import get_logger
from src.paths import reports_dir

log = get_logger(__name__)


def _load_meta(pred_path: Path) -> Dict[str, Any]:
    meta_path = pred_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def evaluate_file(
    pred_path: Path,
    cfg: RunConfig,
    execute: bool = False,
    validate_syntax: bool = False,
    only_db: Optional[str] = None,
    limit: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Scores a single JSONL predictions file across text and execution metrics."""
    rows = read_jsonl(pred_path)
    if limit:
        rows = rows[:limit]
    meta = _load_meta(pred_path)
    log.info("Scoring %s (%d rows)", pred_path.name, len(rows))

    for row in rows:
        score_row(row)

    if execute or validate_syntax:
        executor = Neo4jExecutor(cfg.neo4j, use_cache=use_cache)
        skipped_no_db = 0
        skipped_filtered = 0
        t0 = time.time()

        for i, row in enumerate(rows):
            alias = row.get("database_reference")
            if not alias:
                skipped_no_db += 1
                continue
            if only_db and only_db not in alias:
                skipped_filtered += 1
                continue

            target = resolve_target(alias, cfg.neo4j)
            if target is None:
                skipped_no_db += 1
                continue

            if validate_syntax:
                ok, err = executor.validate_syntax(target, row.get("predicted_cypher", ""))
                row["syntax_valid"] = ok
                row["syntax_error"] = err

            if execute:
                row.update(executor.compare(
                    target,
                    row.get("gold_cypher", ""),
                    row.get("predicted_cypher", ""),
                ))

            if (i + 1) % 50 == 0:
                log.info("  Executed %d/%d (%.1fs elapsed)", i + 1, len(rows), time.time() - t0)

        executor.close()
        log.info("Execution phase completed in %.1fs (skipped: %d without database, %d filtered).",
                 time.time() - t0, skipped_no_db, skipped_filtered)

    report: Dict[str, Any] = {
        "predictions_file": str(pred_path),
        "n_rows": len(rows),
        "meta": meta,
        "overall": summarise(rows),
    }

    # Breakdown by topological complexity
    by_complexity: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("complexity", "unknown")].append(row)
    for label in COMPLEXITY_ORDER:
        if grouped.get(label):
            by_complexity[label] = summarise(grouped[label])
    report["by_complexity"] = by_complexity

    # Breakdown by dataset source
    by_source: Dict[str, Dict[str, float]] = {}
    grouped_src: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped_src[row.get("data_source", "unknown")].append(row)
    for source, subset in sorted(grouped_src.items(), key=lambda kv: -len(kv[1]))[:15]:
        by_source[source] = summarise(subset)
    report["by_data_source"] = by_source

    scored_path = pred_path.with_name(pred_path.stem + ".scored.jsonl")
    write_jsonl(scored_path, rows)
    report["scored_file"] = str(scored_path)

    return report


def render_markdown(reports: List[Dict[str, Any]]) -> str:
    """Renders formatted markdown summaries for evaluation reports."""
    lines: List[str] = ["# Text2Cypher Evaluation Report", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append("")
    header = ("| Run | Model | Schema Mode | Backend | N | GLEU | Exact Match | "
              "Norm. EM | Executable | Exec. Acc. | Prompt Tok (Mean) |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rep in reports:
        m = rep.get("meta", {})
        o = rep.get("overall", {})
        lines.append(
            "| {run} | {model} | {schema} | {backend} | {n} | {gleu} | {em} | {nem} | "
            "{ex} | {acc} | {tok} |".format(
                run=m.get("tag") or m.get("run_name") or Path(rep["predictions_file"]).stem[:24],
                model=m.get("model_key", "?"),
                schema=m.get("schema_mode", "?"),
                backend=m.get("backend", "?"),
                n=o.get("n", 0),
                gleu=o.get("google_bleu_corpus", "-"),
                em=o.get("exact_match", "-"),
                nem=o.get("normalised_exact_match", "-"),
                ex=o.get("executable_rate", "-"),
                acc=o.get("execution_accuracy", "-"),
                tok=o.get("prompt_tokens_mean", "-"),
            )
        )
    lines.append("")

    for rep in reports:
        m = rep.get("meta", {})
        name = m.get("tag") or Path(rep["predictions_file"]).stem
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Predictions File: `{rep['predictions_file']}`")
        if m:
            lines.append(f"- Model: `{m.get('model_key')}` "
                         f"(adapter: `{m.get('adapter') or 'none'}`)")
            lines.append(f"- Backend: `{m.get('backend')}`, "
                         f"Schema Mode: `{m.get('schema_mode')}`")
            if m.get("seconds_per_item"):
                lines.append(f"- Latency: {m['seconds_per_item']} s/item on "
                             f"`{m.get('host', 'unknown host')}`")
        lines.append("")

        lines.append("### By Query Complexity")
        lines.append("")
        lines.append("| Complexity | N | GLEU | Exact Match | Executable | Exec. Acc. |")
        lines.append("|---|---|---|---|---|---|")
        for label in COMPLEXITY_ORDER:
            block = rep.get("by_complexity", {}).get(label)
            if not block:
                continue
            lines.append(
                f"| {label} | {block.get('n', 0)} | "
                f"{block.get('google_bleu_corpus', '-')} | "
                f"{block.get('exact_match', '-')} | "
                f"{block.get('executable_rate', '-')} | "
                f"{block.get('execution_accuracy', '-')} |"
            )
        lines.append("")

        if rep.get("by_data_source"):
            lines.append("### By Data Source")
            lines.append("")
            lines.append("| Source | N | GLEU | Exact Match | Exec. Acc. |")
            lines.append("|---|---|---|---|---|")
            for source, block in rep["by_data_source"].items():
                lines.append(
                    f"| {source} | {block.get('n', 0)} | "
                    f"{block.get('google_bleu_corpus', '-')} | "
                    f"{block.get('exact_match', '-')} | "
                    f"{block.get('execution_accuracy', '-')} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**Definitions:** Google-BLEU (GLEU) measures code-aware n-gram overlap. "
                 "Exact Match validates exact query string equality; Normalised Exact Match "
                 "ignores whitespace and keyword casing. Executable indicates queries running without "
                 "syntax errors. Execution Accuracy validates result set equivalence with the reference query.")
    return "\n".join(lines)


def evaluate_files(
    prediction_files: List[str],
    execute: bool = False,
    validate_syntax: bool = False,
    provider: Optional[str] = None,
    only_db: Optional[str] = None,
    limit: Optional[int] = None,
    out_path: Optional[str] = None,
    use_cache: bool = True,
) -> Path:
    cfg = RunConfig()
    if provider:
        cfg.neo4j.provider = provider

    reports: List[Dict[str, Any]] = []
    for raw_path in prediction_files:
        path = Path(raw_path)
        if not path.exists():
            log.error("Prediction file not found: %s", path)
            continue
        reports.append(evaluate_file(
            path, cfg,
            execute=execute,
            validate_syntax=validate_syntax,
            only_db=only_db,
            limit=limit,
            use_cache=use_cache,
        ))

    if not reports:
        raise SystemExit("No prediction files could be evaluated.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_path) if out_path else reports_dir() / f"evaluation-{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(reports)
    out.write_text(markdown, encoding="utf-8")

    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")

    print()
    print(markdown)
    print()
    log.info("Evaluation report saved to %s (and %s)", out, json_out)
    return out
