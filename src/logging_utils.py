"""Minimal, dependency-free logging setup shared by every entrypoint."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_LEVEL_COLOURS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class _Formatter(logging.Formatter):
    def __init__(self, colour: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.colour:
            return text
        prefix = _LEVEL_COLOURS.get(record.levelname, "")
        if not prefix:
            return text
        return f"{prefix}{text}{_RESET}"


def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_Formatter(colour=_supports_colour()))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # These libraries are extremely chatty at INFO.
    for noisy in ("urllib3", "filelock", "neo4j", "httpx", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
