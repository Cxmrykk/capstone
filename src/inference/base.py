"""Backend interface.

Both backends receive fully-rendered prompt strings produced by
`src.data.prompts.render_prompt`. Prompt construction therefore lives in
exactly one place, so a Colab result and a laptop GGUF result are comparable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GenerationBackend(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def generate(self, prompts: List[str], max_new_tokens: int = 256,
                 temperature: float = 0.0, **kwargs) -> List[str]:
        ...

    def tokenizer(self):
        """The tokenizer used for prompt rendering (may be None)."""
        return None

    def info(self) -> Dict[str, Any]:
        return {"backend": self.name}

    def close(self) -> None:
        return None

    def __enter__(self) -> "GenerationBackend":
        self.load()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
