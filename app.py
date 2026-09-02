#!/usr/bin/env python3
"""
Text2Cypher toolkit: fine-tuning and evaluating lightweight LLMs for graph databases.

Usage:
    python app.py doctor
    python app.py data stats
    python app.py train --config configs/qwen3.5-2b.yaml
    python app.py predict --config configs/qwen3.5-2b.yaml --adapter artifacts/runs/<run>/final
    python app.py evaluate --predictions artifacts/predictions/<file>.jsonl --execute
    python app.py export merge --config ... --adapter ...
    python app.py export gguf --merged artifacts/merged/<name> --quant Q4_K_M
    python app.py checkpoint list --config ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `src` importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.py",
        description="Text2Cypher toolkit: fine-tuning and evaluating lightweight LLMs for graph databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    p.add_argument("--quiet", action="store_true", help="Warnings and errors only.")

    sub = p.add_subparsers(dest="command", required=True)

    # ---- doctor ----------------------------------------------------------- #
    d = sub.add_parser("doctor", help="Report on the local/Colab environment.")
    d.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    d.add_argument("--check-neo4j", action="store_true", help="Also probe Neo4j connectivity.")

    # ---- data ------------------------------------------------------------- #
    data = sub.add_parser("data", help="Dataset inspection and preparation.")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    ds_stats = data_sub.add_parser("stats", help="Summarise the dataset splits.")
    ds_stats.add_argument("--dataset-dir", default=None)

    ds_prev = data_sub.add_parser("preview", help="Render a fully-formatted training example.")
    ds_prev.add_argument("--config", default=None)
    ds_prev.add_argument("--index", type=int, default=0)
    ds_prev.add_argument("--split", default="train", choices=["train", "test"])
    ds_prev.add_argument("--schema-mode", default=None)
    ds_prev.add_argument("--no-tokenizer", action="store_true",
                         help="Skip loading the tokenizer (shows the raw prompt body only).")

    ds_tok = data_sub.add_parser("token-stats", help="Token-length distribution for a schema mode.")
    ds_tok.add_argument("--config", required=True)
    ds_tok.add_argument("--schema-mode", default=None)
    ds_tok.add_argument("--samples", type=int, default=1000)

    # ---- train ------------------------------------------------------------ #
    t = sub.add_parser("train", help="LoRA fine-tuning with Hub checkpoint sync.")
    t.add_argument("--config", required=True)
    t.add_argument("--run-name", default=None)
    t.add_argument("--resume", default="auto", choices=["auto", "hub", "local", "none"])
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--time-limit", type=float, default=None,
                   help="Stop and checkpoint after N minutes (useful for Colab session budgets).")
    t.add_argument("--schema-mode", default=None)
    t.add_argument("--max-train-samples", type=int, default=None)
    t.add_argument("--hub-repo", default=None)
    t.add_argument("--no-hub", action="store_true", help="Disable Hub checkpoint sync.")
    t.add_argument("--output-dir", default=None)

    # ---- predict ---------------------------------------------------------- #
    pr = sub.add_parser("predict", help="Generate Cypher for an evaluation split.")
    pr.add_argument("--config", required=True)
    pr.add_argument("--backend", default="hf", choices=["hf", "llamacpp"])
    pr.add_argument("--adapter", default=None, help="Path to a LoRA adapter directory.")
    pr.add_argument("--model-path", default=None, help="Override base/merged model path.")
    pr.add_argument("--gguf", default=None, help="GGUF file (llamacpp backend).")
    pr.add_argument("--server-url", default=None, help="Existing llama-server URL.")
    pr.add_argument("--split", default="test", choices=["train", "test"])
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--schema-mode", default=None)
    pr.add_argument("--batch-size", type=int, default=None)
    pr.add_argument("--max-new-tokens", type=int, default=None)
    pr.add_argument("--out", default=None)
    pr.add_argument("--tag", default=None, help="Label recorded in the output file.")

    # ---- evaluate --------------------------------------------------------- #
    e = sub.add_parser("evaluate", help="Score a predictions file.")
    e.add_argument("--predictions", required=True, nargs="+")
    e.add_argument("--execute", action="store_true", help="Run execution-based evaluation.")
    e.add_argument("--validate-syntax", action="store_true", help="EXPLAIN every query.")
    e.add_argument("--provider", default=None, choices=["demo", "local"])
    e.add_argument("--only-db", default=None, help="Restrict execution to one database alias.")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--out", default=None)
    e.add_argument("--no-cache", action="store_true")

    # ---- export ----------------------------------------------------------- #
    x = sub.add_parser("export", help="Merge adapters and convert to GGUF.")
    x_sub = x.add_subparsers(dest="export_command", required=True)

    xm = x_sub.add_parser("merge", help="Merge a LoRA adapter into fp16 base weights.")
    xm.add_argument("--config", required=True)
    xm.add_argument("--adapter", required=True)
    xm.add_argument("--out", default=None)
    xm.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    xg = x_sub.add_parser("gguf", help="Convert merged weights to GGUF and quantise.")
    xg.add_argument("--merged", required=True)
    xg.add_argument("--out-dir", default=None)
    xg.add_argument("--quant", default="Q4_K_M")
    xg.add_argument("--llama-cpp", default=None, help="Path to a llama.cpp checkout.")
    xg.add_argument("--keep-f16", action="store_true", help="Keep the intermediate f16 GGUF.")

    # ---- checkpoint ------------------------------------------------------- #
    c = sub.add_parser("checkpoint", help="Inspect/move checkpoints on the Hub.")
    c_sub = c.add_subparsers(dest="checkpoint_command", required=True)

    cl = c_sub.add_parser("list", help="List remote checkpoints for a run.")
    cl.add_argument("--config", required=True)
    cl.add_argument("--run-name", default=None)

    cp = c_sub.add_parser("pull", help="Download a checkpoint from the Hub.")
    cp.add_argument("--config", required=True)
    cp.add_argument("--run-name", default=None)
    cp.add_argument("--step", default="latest")
    cp.add_argument("--dest", default=None)

    cu = c_sub.add_parser("push", help="Upload a local checkpoint directory.")
    cu.add_argument("--config", required=True)
    cu.add_argument("--path", required=True)
    cu.add_argument("--run-name", default=None)

    # ---- neo4j ------------------------------------------------------------ #
    n = sub.add_parser("neo4j", help="Neo4j connectivity helpers.")
    n_sub = n.add_subparsers(dest="neo4j_command", required=True)

    nc = n_sub.add_parser("check", help="Probe a database alias.")
    nc.add_argument("--alias", default="neo4jlabs_demo_db_movies")
    nc.add_argument("--provider", default=None, choices=["demo", "local"])

    nl = n_sub.add_parser("aliases", help="List database aliases present in the dataset.")
    nl.add_argument("--dataset-dir", default=None)

    return p


# --------------------------------------------------------------------------- #
# Command dispatch
# --------------------------------------------------------------------------- #
def cmd_doctor(args) -> int:
    from src.env import environment_report, render_report
    import json as _json

    report = environment_report(check_neo4j=args.check_neo4j)
    if args.json:
        print(_json.dumps(report, indent=2, default=str))
    else:
        print(render_report(report))
    return 0


def cmd_data(args) -> int:
    from src.config import RunConfig, load_config
    from src.data.dataset import dataset_stats, preview_example, token_length_stats

    if args.data_command == "stats":
        print(dataset_stats(args.dataset_dir))
        return 0

    if args.data_command == "preview":
        cfg = load_config(args.config) if args.config else RunConfig()
        if args.schema_mode:
            cfg.data.schema_mode = args.schema_mode
        print(preview_example(cfg, index=args.index, split=args.split,
                              use_tokenizer=not args.no_tokenizer))
        return 0

    if args.data_command == "token-stats":
        cfg = load_config(args.config)
        if args.schema_mode:
            cfg.data.schema_mode = args.schema_mode
        print(token_length_stats(cfg, n_samples=args.samples))
        return 0

    raise SystemExit(f"Unknown data command: {args.data_command}")


def cmd_train(args) -> int:
    from src.config import load_config
    from src.training.train import run_training

    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.schema_mode:
        cfg.data.schema_mode = args.schema_mode
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.time_limit is not None:
        cfg.train.time_limit_minutes = args.time_limit
    if args.max_train_samples is not None:
        cfg.data.max_train_samples = args.max_train_samples
    if args.hub_repo:
        cfg.hub.repo_id = args.hub_repo
    if args.no_hub:
        cfg.hub.sync = False
    if args.output_dir:
        cfg.output_dir = args.output_dir

    run_training(cfg, resume=args.resume)
    return 0


def cmd_predict(args) -> int:
    from src.config import load_config
    from src.inference.predict import run_prediction

    cfg = load_config(args.config)
    if args.schema_mode:
        cfg.data.schema_mode = args.schema_mode
    if args.batch_size is not None:
        cfg.generation.batch_size = args.batch_size
    if args.max_new_tokens is not None:
        cfg.generation.max_new_tokens = args.max_new_tokens

    out = run_prediction(
        cfg,
        backend_name=args.backend,
        adapter=args.adapter,
        model_path=args.model_path,
        gguf=args.gguf,
        server_url=args.server_url,
        split=args.split,
        limit=args.limit,
        out_path=args.out,
        tag=args.tag,
    )
    log.info("Predictions written to %s", out)
    return 0


def cmd_evaluate(args) -> int:
    from src.evaluation.evaluate import evaluate_files

    evaluate_files(
        prediction_files=args.predictions,
        execute=args.execute,
        validate_syntax=args.validate_syntax,
        provider=args.provider,
        only_db=args.only_db,
        limit=args.limit,
        out_path=args.out,
        use_cache=not args.no_cache,
    )
    return 0


def cmd_export(args) -> int:
    from src.config import load_config

    if args.export_command == "merge":
        from src.export.merge import merge_adapter

        cfg = load_config(args.config)
        out = merge_adapter(cfg, adapter_path=args.adapter, out_dir=args.out, dtype=args.dtype)
        log.info("Merged model written to %s", out)
        return 0

    if args.export_command == "gguf":
        from src.export.gguf import convert_to_gguf

        out = convert_to_gguf(
            merged_dir=args.merged,
            out_dir=args.out_dir,
            quant=args.quant,
            llama_cpp_dir=args.llama_cpp,
            keep_f16=args.keep_f16,
        )
        log.info("GGUF written to %s", out)
        return 0

    raise SystemExit(f"Unknown export command: {args.export_command}")


def cmd_checkpoint(args) -> int:
    from src.config import load_config
    from src.training.checkpointing import HubCheckpointSync

    cfg = load_config(args.config)
    run_name = getattr(args, "run_name", None) or cfg.run_name
    sync = HubCheckpointSync.from_config(cfg, run_name=run_name)

    if args.checkpoint_command == "list":
        entries = sync.list_checkpoints()
        if not entries:
            print(f"No remote checkpoints found for run '{run_name}' in {sync.repo_id}.")
            return 0
        print(f"Remote checkpoints for '{run_name}' in {sync.repo_id}:")
        for step in entries:
            print(f"  checkpoint-{step}")
        latest = sync.read_latest_marker()
        if latest:
            print(f"\nLATEST marker -> step {latest.get('step')} "
                  f"(written {latest.get('written_at')})")
        return 0

    if args.checkpoint_command == "pull":
        step = None if args.step == "latest" else int(args.step)
        dest = sync.pull(step=step, dest=args.dest)
        print(f"Checkpoint downloaded to {dest}")
        return 0

    if args.checkpoint_command == "push":
        path = Path(args.path)
        step = _infer_step_from_dirname(path)
        sync.push(path, step=step, blocking=True)
        print(f"Uploaded {path} as checkpoint-{step}")
        return 0

    raise SystemExit(f"Unknown checkpoint command: {args.checkpoint_command}")


def _infer_step_from_dirname(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except ValueError:
            pass
    return 0


def cmd_neo4j(args) -> int:
    from src.config import RunConfig
    from src.evaluation.neo4j_client import Neo4jExecutor, resolve_target
    from src.data.dataset import list_database_aliases

    if args.neo4j_command == "aliases":
        aliases = list_database_aliases(args.dataset_dir)
        for alias, count in aliases:
            print(f"{count:>7}  {alias}")
        return 0

    if args.neo4j_command == "check":
        cfg = RunConfig()
        if args.provider:
            cfg.neo4j.provider = args.provider
        target = resolve_target(args.alias, cfg.neo4j)
        if target is None:
            print(f"Could not resolve alias '{args.alias}' under provider "
                  f"'{cfg.neo4j.provider}'.")
            return 1
        print(f"Resolved -> uri={target.uri} database={target.database} user={target.user}")
        executor = Neo4jExecutor(cfg.neo4j, use_cache=False)
        ok, detail = executor.ping(target)
        executor.close()
        print("Reachable" if ok else f"Unreachable: {detail}")
        return 0 if ok else 1

    raise SystemExit(f"Unknown neo4j command: {args.neo4j_command}")


DISPATCH = {
    "doctor": cmd_doctor,
    "data": cmd_data,
    "train": cmd_train,
    "predict": cmd_predict,
    "evaluate": cmd_evaluate,
    "export": cmd_export,
    "checkpoint": cmd_checkpoint,
    "neo4j": cmd_neo4j,
}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO")
    setup_logging(level)

    # Quieter, more predictable HF behaviour by default.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    handler = DISPATCH[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
