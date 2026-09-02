"""llama.cpp generation backend for local and CPU inference on GGUF models.

Drives `llama-server` over local HTTP endpoints to ensure cross-platform compatibility
without requiring C Python bindings, while maintaining identical prompt formatting.
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import RunConfig
from src.inference.base import GenerationBackend
from src.logging_utils import get_logger

log = get_logger(__name__)

_DEFAULT_PORT = 8757


def find_llama_binary(name: str = "llama-server",
                      llama_cpp_dir: Optional[str] = None) -> Optional[str]:
    """Locates a compiled llama.cpp binary on PATH or within known build locations."""
    on_path = shutil.which(name)
    if on_path:
        return on_path

    roots: List[Path] = []
    for candidate in (llama_cpp_dir, os.environ.get("LLAMA_CPP_DIR")):
        if candidate:
            roots.append(Path(candidate))
    roots += [Path.home() / "llama.cpp", Path("/opt/llama.cpp")]
    try:
        from src.paths import repo_root

        roots.append(repo_root() / "vendor" / "llama.cpp")
    except Exception:
        pass

    for root in roots:
        for rel in ("build/bin", "build", "bin", "."):
            candidate = root / rel / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


class LlamaCppBackend(GenerationBackend):
    name = "llamacpp"

    def __init__(
        self,
        cfg: RunConfig,
        gguf_path: Optional[str] = None,
        server_url: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        port: int = _DEFAULT_PORT,
        llama_cpp_dir: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        startup_timeout_s: float = 240.0,
    ) -> None:
        self.cfg = cfg
        self.gguf_path = gguf_path
        self.server_url = server_url.rstrip("/") if server_url else None
        self.n_ctx = n_ctx or cfg.data.max_seq_length + cfg.generation.max_new_tokens + 64
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.port = port
        self.llama_cpp_dir = llama_cpp_dir
        self.tokenizer_path = tokenizer_path
        self.startup_timeout_s = startup_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._tokenizer = None
        self._owns_server = False

    def load(self) -> None:
        self._load_tokenizer()
        if self.server_url:
            if not self._wait_healthy(self.server_url, timeout=15.0):
                raise RuntimeError(f"llama-server at {self.server_url} is unreachable.")
            log.info("Connected to existing llama-server at %s", self.server_url)
            return
        self._spawn_server()

    def _load_tokenizer(self) -> None:
        from src.training.model_loader import load_tokenizer

        cfg = self.cfg
        if self.tokenizer_path:
            original = cfg.model.path
            cfg.model.path = self.tokenizer_path
            try:
                self._tokenizer = load_tokenizer(cfg, padding_side="left")
            finally:
                cfg.model.path = original
        else:
            self._tokenizer = load_tokenizer(cfg, padding_side="left")

    def _spawn_server(self) -> None:
        if not self.gguf_path:
            raise ValueError("LlamaCppBackend requires either --gguf or --server-url.")
        gguf = Path(self.gguf_path)
        if not gguf.exists():
            raise FileNotFoundError(f"GGUF model file not found: {gguf}")

        binary = find_llama_binary("llama-server", self.llama_cpp_dir)
        if not binary:
            raise RuntimeError(
                "llama-server executable not found. Build llama.cpp with:\n"
                "  bash scripts/build_llama_cpp.sh\n"
                "or set LLAMA_CPP_DIR to an existing installation."
            )

        cmd = [
            binary,
            "-m", str(gguf),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--parallel", "1",
            "--no-warmup",
        ]
        log.info("Starting llama-server process: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        self._owns_server = True
        atexit.register(self.close)

        url = f"http://127.0.0.1:{self.port}"
        if not self._wait_healthy(url, self.startup_timeout_s):
            self.close()
            raise RuntimeError(
                f"llama-server failed to initialize within {self.startup_timeout_s:.0f}s. "
                "Verify that the model architecture is supported by the current llama.cpp build."
            )
        self.server_url = url
        log.info("llama-server ready at %s (context=%d, threads=%d)",
                 url, self.n_ctx, self.n_threads)

    @staticmethod
    def _wait_healthy(url: str, timeout: float) -> bool:
        import requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{url}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1.5)
        return False

    def tokenizer(self):
        return self._tokenizer

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "gguf": self.gguf_path,
            "server_url": self.server_url,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
        }

    def generate(self, prompts: List[str], max_new_tokens: int = 256,
                 temperature: float = 0.0, **kwargs) -> List[str]:
        import requests

        if self.server_url is None:
            self.load()

        stop = list(self.cfg.generation.stop_strings)
        eos = getattr(self._tokenizer, "eos_token", None)
        if eos and eos not in stop:
            stop.append(eos)

        outputs: List[str] = []
        t0 = time.time()
        session = requests.Session()

        for i, prompt in enumerate(prompts):
            payload = {
                "prompt": prompt,
                "n_predict": max_new_tokens,
                "temperature": float(temperature),
                "top_p": self.cfg.generation.top_p,
                "stop": stop,
                "cache_prompt": True,
                "stream": False,
            }
            if temperature <= 0:
                payload["top_k"] = 1

            text = ""
            for attempt in range(3):
                try:
                    resp = session.post(f"{self.server_url}/completion",
                                        json=payload, timeout=600)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("content", "")
                    break
                except Exception as exc:
                    log.warning("llama-server request error (attempt %d/3): %s",
                                attempt + 1, exc)
                    time.sleep(2 * (attempt + 1))
            outputs.append(text)

            if (i + 1) % 20 == 0:
                rate = (i + 1) / max(1e-6, time.time() - t0)
                log.info("Generated %d/%d (%.2f items/s)", i + 1, len(prompts), rate)

        log.info("Generation complete: %d prompts in %.1fs.", len(prompts), time.time() - t0)
        return outputs

    def close(self) -> None:
        if self._proc is not None and self._owns_server:
            log.info("Shutting down llama-server (PID %s).", self._proc.pid)
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                else:
                    self._proc.terminate()
                self._proc.wait(timeout=15)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
