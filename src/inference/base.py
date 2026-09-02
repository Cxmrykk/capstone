"""Abstract base interface for generation backends.

Provides a unified interface for prompt-based generation across GPU Transformers
and GGUF/llama.cpp inference engines, ensuring identical prompt formatting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GenerationBackend(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Loads and initializes model weights and resources."""
        ...

    @abstractmethod
    def generate(self, prompts: List[str], max_new_tokens: int = 256,
                 temperature: float = 0.0, **kwargs) -> List[str]:
        """Generates completions for a batch of pre-formatted prompt strings."""
        ...

    def tokenizer(self):
        """Returns the associated tokenizer used for prompt formatting."""
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
