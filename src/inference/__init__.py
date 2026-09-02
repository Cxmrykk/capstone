"""Generation backends: HF transformers (GPU) and llama.cpp (CPU/edge)."""

from src.inference.base import GenerationBackend

__all__ = ["GenerationBackend", "get_backend"]


def get_backend(name: str, **kwargs) -> GenerationBackend:
    if name == "hf":
        from src.inference.hf_backend import TransformersBackend

        return TransformersBackend(**kwargs)
    if name == "llamacpp":
        from src.inference.llama_cpp_backend import LlamaCppBackend

        return LlamaCppBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r} (expected 'hf' or 'llamacpp')")
