"""Hub-backed checkpoint continuity for account-cycled Colab sessions.

The problem: a free-tier session can vanish at any moment, and the next session
may be on a different Google account with a fresh filesystem. The fix is to
treat a private HF model repo as the source of truth for training state.

What gets uploaded on every save:
    adapter_model.safetensors, adapter_config.json   (LoRA weights)
    optimizer.pt, scheduler.pt, rng_state*.pth       (exact resume)
    trainer_state.json, training_args.bin            (step/epoch bookkeeping)

Because only LoRA parameters are trainable, the optimizer state is small, so a
full checkpoint is typically tens to a few hundred MB rather than tens of GB.

Atomicity: the LATEST.json marker is written *after* the folder upload
succeeds, so a session killed mid-upload leaves the previous marker intact and
resume falls back to the last complete checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import RunConfig
from src.logging_utils import get_logger
from src.paths import artifacts_dir, ensure_dir

log = get_logger(__name__)

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

# Never ship full base weights to the checkpoint repo; only adapters + state.
_IGNORE_PATTERNS = [
    "*.gguf",
    "global_step*/*",
    "*.pt.tmp",
]


def _hf_token(cfg_env: str = "HF_TOKEN") -> Optional[str]:
    return (
        os.environ.get(cfg_env)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


@dataclass
class HubCheckpointSync:
    repo_id: str
    run_name: str
    token: Optional[str] = None
    private: bool = True
    keep_last: int = 2
    enabled: bool = True
    async_upload: bool = True
    drive_mirror: Optional[str] = None

    _thread: Optional[threading.Thread] = None
    _lock: Optional[threading.Lock] = None
    _repo_ready: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if self.token is None:
            self.token = _hf_token()

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_config(cls, cfg: RunConfig, run_name: Optional[str] = None) -> "HubCheckpointSync":
        repo_id = cfg.hub.repo_id or os.environ.get("T2C_HUB_REPO")
        if not repo_id:
            raise ValueError(
                "No checkpoint repository configured. Set hub.repo_id in the config "
                "(e.g. 'your-username/t2c-capstone-checkpoints') or export T2C_HUB_REPO."
            )
        return cls(
            repo_id=repo_id,
            run_name=run_name or cfg.run_name,
            token=_hf_token(cfg.hub.token_env),
            private=cfg.hub.private,
            keep_last=cfg.hub.keep_last,
            enabled=cfg.hub.sync,
            async_upload=cfg.hub.async_upload,
            drive_mirror=cfg.hub.drive_mirror,
        )

    # -- paths within the repo -------------------------------------------- #
    @property
    def run_prefix(self) -> str:
        return f"runs/{self.run_name}"

    def _ckpt_prefix(self, step: int) -> str:
        return f"{self.run_prefix}/checkpoint-{step}"

    @property
    def _latest_path(self) -> str:
        return f"{self.run_prefix}/LATEST.json"

    # -- api -------------------------------------------------------------- #
    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def ensure_repo(self) -> None:
        if self._repo_ready or not self.enabled:
            return
        if not self.token:
            raise RuntimeError(
                "No Hugging Face token found. Export HF_TOKEN (a write token) before "
                "training, or pass --no-hub to disable checkpoint sync."
            )
        api = self._api()
        api.create_repo(repo_id=self.repo_id, repo_type="model",
                        private=self.private, exist_ok=True)
        self._repo_ready = True
        log.info("Checkpoint repo ready: %s (private=%s, run=%s)",
                 self.repo_id, self.private, self.run_name)

    # -- upload ----------------------------------------------------------- #
    def push(self, local_dir: str | Path, step: int, blocking: bool = False,
             extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        local_dir = Path(local_dir)
        if not local_dir.exists():
            log.warning("Checkpoint directory %s does not exist; skipping upload.", local_dir)
            return

        self.ensure_repo()

        if self.drive_mirror:
            self._mirror_to_drive(local_dir, step)

        if blocking or not self.async_upload:
            self._do_push(local_dir, step, extra_metadata)
            return

        # Serialise uploads: wait for any in-flight upload before starting a new
        # one, so we never interleave commits to the same repo.
        self.wait()
        thread = threading.Thread(
            target=self._do_push,
            args=(local_dir, step, extra_metadata),
            name=f"hub-upload-{step}",
            daemon=False,   # non-daemon: let it finish if the process exits cleanly
        )
        self._thread = thread
        thread.start()

    def _do_push(self, local_dir: Path, step: int,
                 extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            start = time.time()
            try:
                api = self._api()
                size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e6
                log.info("Uploading checkpoint-%d (%.1f MB) to %s ...",
                         step, size_mb, self.repo_id)

                api.upload_folder(
                    folder_path=str(local_dir),
                    path_in_repo=self._ckpt_prefix(step),
                    repo_id=self.repo_id,
                    repo_type="model",
                    ignore_patterns=_IGNORE_PATTERNS,
                    commit_message=f"[{self.run_name}] checkpoint-{step}",
                )

                # Marker written only after a successful folder commit.
                marker = {
                    "run_name": self.run_name,
                    "step": step,
                    "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "path": self._ckpt_prefix(step),
                }
                if extra_metadata:
                    marker.update(extra_metadata)

                api.upload_file(
                    path_or_fileobj=json.dumps(marker, indent=2).encode("utf-8"),
                    path_in_repo=self._latest_path,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"[{self.run_name}] mark checkpoint-{step} as latest",
                )

                log.info("Uploaded checkpoint-%d in %.1fs.", step, time.time() - start)
                self.prune(keep_last=self.keep_last, protect_step=step)

            except Exception as exc:
                # Never kill a training run over a failed upload.
                log.error("Checkpoint upload for step %d failed: %s", step, exc)

    def _mirror_to_drive(self, local_dir: Path, step: int) -> None:
        try:
            dest = Path(self.drive_mirror) / self.run_name / f"checkpoint-{step}"
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local_dir, dest)
            log.info("Mirrored checkpoint-%d to %s", step, dest)
        except Exception as exc:
            log.warning("Drive mirror failed: %s", exc)

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            log.info("Waiting for in-flight checkpoint upload ...")
            self._thread.join(timeout)
        self._thread = None

    # -- listing / download ------------------------------------------------ #
    def list_checkpoints(self) -> List[int]:
        try:
            api = self._api()
            files = api.list_repo_files(repo_id=self.repo_id, repo_type="model")
        except Exception as exc:
            log.warning("Could not list %s: %s", self.repo_id, exc)
            return []

        steps = set()
        prefix = f"{self.run_prefix}/"
        for f in files:
            if not f.startswith(prefix):
                continue
            m = _CHECKPOINT_RE.search(f[len(prefix):])
            if m:
                steps.add(int(m.group(1)))
        return sorted(steps)

    def read_latest_marker(self) -> Optional[Dict[str, Any]]:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self._latest_path,
                repo_type="model",
                token=self.token,
            )
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            log.debug("No LATEST marker for run %s: %s", self.run_name, exc)
            return None

    def latest_step(self) -> Optional[int]:
        marker = self.read_latest_marker()
        if marker and "step" in marker:
            return int(marker["step"])
        steps = self.list_checkpoints()
        return steps[-1] if steps else None

    def pull(self, step: Optional[int] = None, dest: Optional[str | Path] = None
             ) -> Optional[Path]:
        from huggingface_hub import snapshot_download

        if step is None:
            step = self.latest_step()
        if step is None:
            log.info("No remote checkpoint available for run '%s'.", self.run_name)
            return None

        dest_root = Path(dest) if dest else (artifacts_dir() / "hub_pull" / self.run_name)
        ensure_dir(dest_root)

        log.info("Downloading %s/checkpoint-%d from %s ...", self.run_prefix, step, self.repo_id)
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
            allow_patterns=[f"{self._ckpt_prefix(step)}/*"],
            local_dir=str(dest_root),
        )

        # snapshot_download preserves the repo layout underneath dest_root.
        resolved = dest_root / self.run_prefix / f"checkpoint-{step}"
        if not resolved.exists():
            log.error("Expected %s after download but it is missing.", resolved)
            return None
        log.info("Checkpoint available at %s", resolved)
        return resolved

    def prune(self, keep_last: int, protect_step: Optional[int] = None) -> None:
        if keep_last <= 0:
            return
        steps = self.list_checkpoints()
        if len(steps) <= keep_last:
            return
        doomed = steps[:-keep_last]
        if protect_step is not None:
            doomed = [s for s in doomed if s != protect_step]
        if not doomed:
            return

        api = self._api()
        for step in doomed:
            try:
                api.delete_folder(
                    path_in_repo=self._ckpt_prefix(step),
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"[{self.run_name}] prune checkpoint-{step}",
                )
                log.info("Pruned remote checkpoint-%d.", step)
            except Exception as exc:
                log.warning("Could not prune checkpoint-%d: %s", step, exc)

    def push_final(self, local_dir: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Upload the finished adapter to runs/<name>/final."""
        if not self.enabled:
            return
        self.ensure_repo()
        try:
            api = self._api()
            api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=f"{self.run_prefix}/final",
                repo_id=self.repo_id,
                repo_type="model",
                ignore_patterns=_IGNORE_PATTERNS,
                commit_message=f"[{self.run_name}] final adapter",
            )
            if metadata:
                api.upload_file(
                    path_or_fileobj=json.dumps(metadata, indent=2, default=str).encode("utf-8"),
                    path_in_repo=f"{self.run_prefix}/final/run_metadata.json",
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"[{self.run_name}] run metadata",
                )
            log.info("Final adapter uploaded to %s/%s/final", self.repo_id, self.run_prefix)
        except Exception as exc:
            log.error("Final upload failed: %s", exc)


# --------------------------------------------------------------------------- #
# Trainer integration
# --------------------------------------------------------------------------- #
def build_callbacks(cfg: RunConfig, sync: Optional[HubCheckpointSync],
                    fingerprint: str):
    """Assemble the callback list. Imported lazily to keep transformers optional."""
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    callbacks = []

    class HubCheckpointCallback(TrainerCallback):
        """Uploads each local checkpoint to the Hub as soon as it is written."""

        def on_save(self, args, state, control, **kwargs):
            if sync is None or not sync.enabled or not state.is_world_process_zero:
                return control
            ckpt_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            sync.push(
                ckpt_dir,
                step=state.global_step,
                extra_metadata={
                    "config_fingerprint": fingerprint,
                    "epoch": state.epoch,
                    "model_key": cfg.model.key,
                    "schema_mode": cfg.data.schema_mode,
                },
            )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if sync is not None:
                sync.wait()
            return control

    class WallClockLimitCallback(TrainerCallback):
        """Stops training cleanly before a Colab session is likely to be reclaimed."""

        def __init__(self, minutes: float) -> None:
            self.limit_s = minutes * 60.0
            self.start = time.time()
            self.tripped = False

        def on_step_end(self, args, state, control, **kwargs):
            if self.tripped:
                return control
            elapsed = time.time() - self.start
            if elapsed >= self.limit_s:
                self.tripped = True
                log.warning(
                    "Wall-clock limit of %.0f min reached at step %d. "
                    "Saving a checkpoint and stopping; resume with --resume hub.",
                    self.limit_s / 60, state.global_step,
                )
                control.should_save = True
                control.should_training_stop = True
            return control

    class JsonlLossLogger(TrainerCallback):
        """Appends every log line to metrics.jsonl for later plotting."""

        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.is_world_process_zero:
                return control
            row = dict(logs)
            row["step"] = state.global_step
            row["epoch"] = state.epoch
            row["wall_time"] = time.time()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            return control

    class VramLogger(TrainerCallback):
        """Records peak VRAM -- needed for the 'constrained hardware' claims."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                import torch
                if torch.cuda.is_available() and logs is not None:
                    logs["vram_peak_gb"] = round(
                        torch.cuda.max_memory_allocated() / (1024 ** 3), 3
                    )
            except Exception:
                pass
            return control

    if sync is not None and sync.enabled:
        callbacks.append(HubCheckpointCallback())
    if cfg.train.time_limit_minutes:
        callbacks.append(WallClockLimitCallback(cfg.train.time_limit_minutes))
    callbacks.append(JsonlLossLogger(cfg.resolved_output_dir() / "metrics.jsonl"))
    callbacks.append(VramLogger())
    return callbacks


# --------------------------------------------------------------------------- #
# Resume resolution
# --------------------------------------------------------------------------- #
def find_local_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        m = _CHECKPOINT_RE.fullmatch(child.name)
        if m and (child / "trainer_state.json").exists():
            candidates.append((int(m.group(1)), child))
    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[0])[1]


def resolve_resume(cfg: RunConfig, mode: str,
                   sync: Optional[HubCheckpointSync]) -> Optional[Path]:
    """Return a checkpoint directory to resume from, or None for a fresh start."""
    if mode == "none":
        return None

    output_dir = cfg.resolved_output_dir()
    fingerprint = cfg.fingerprint()

    if mode in {"auto", "local"}:
        local = find_local_checkpoint(output_dir)
        if local is not None:
            log.info("Resuming from local checkpoint %s", local)
            return local
        if mode == "local":
            log.info("No local checkpoint found in %s; starting fresh.", output_dir)
            return None

    if mode in {"auto", "hub"}:
        if sync is None or not sync.enabled:
            log.info("Hub sync disabled; starting fresh.")
            return None
        marker = sync.read_latest_marker()
        if marker:
            remote_fp = marker.get("config_fingerprint")
            if remote_fp and remote_fp != fingerprint:
                log.warning(
                    "Remote checkpoint fingerprint %s != current config fingerprint %s. "
                    "The config has changed since that checkpoint was written. "
                    "Refusing to resume -- rename the run or pass --resume none.",
                    remote_fp, fingerprint,
                )
                raise SystemExit(2)
        pulled = sync.pull()
        if pulled is None:
            log.info("No remote checkpoint for run '%s'; starting fresh.", cfg.run_name)
            return None

        # Copy into output_dir so Trainer's own bookkeeping stays consistent.
        step = int(_CHECKPOINT_RE.search(pulled.name).group(1))
        dest = output_dir / f"checkpoint-{step}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pulled, dest)
        log.info("Resuming from Hub checkpoint at step %d (%s)", step, dest)
        return dest

    return None
