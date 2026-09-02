"""Transformers and PyTorch generation backend for GPU and CPU execution."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.config import RunConfig
from src.inference.base import GenerationBackend
from src.logging_utils import get_logger

log = get_logger(__name__)


class TransformersBackend(GenerationBackend):
    name = "hf"

    def __init__(self, cfg: RunConfig, adapter: Optional[str] = None,
                 model_path: Optional[str] = None) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.model_path = model_path
        self.model = None
        self._tokenizer = None
        self._meta: Dict[str, Any] = {}

    def load(self) -> None:
        from src.training.model_loader import load_model_and_tokenizer

        model, tokenizer, meta = load_model_and_tokenizer(
            self.cfg,
            for_training=False,
            adapter_path=self.adapter,
            model_path_override=self.model_path,
        )
        model.eval()
        self.model = model
        self._tokenizer = tokenizer
        self._meta = meta
        # Left-padding is required for batched autoregressive decoder generation
        self._tokenizer.padding_side = "left"
        log.info("Transformers backend ready (adapter=%s).", self.adapter or "none")

    def tokenizer(self):
        return self._tokenizer

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "model_path": self.model_path or self.cfg.model.resolved_path(),
            "adapter": self.adapter,
            **self._meta,
        }

    def generate(self, prompts: List[str], max_new_tokens: int = 256,
                 temperature: float = 0.0, batch_size: Optional[int] = None,
                 **kwargs) -> List[str]:
        import torch

        if self.model is None:
            self.load()

        batch_size = batch_size or self.cfg.generation.batch_size
        tok = self._tokenizer
        device = next(self.model.parameters()).device

        # Sort prompts by length to minimize padding overhead within batches
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs: List[str] = [""] * len(prompts)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=self.cfg.generation.top_p,
            )
        else:
            gen_kwargs.update(do_sample=False)
        if self.cfg.generation.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.cfg.generation.repetition_penalty

        t0 = time.time()
        done = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch = [prompts[i] for i in idxs]

            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)

            with torch.inference_mode():
                generated = self.model.generate(**enc, **gen_kwargs)

            prompt_len = enc["input_ids"].shape[1]
            for row, idx in zip(generated, idxs):
                new_tokens = row[prompt_len:]
                outputs[idx] = tok.decode(new_tokens, skip_special_tokens=True)

            done += len(idxs)
            if done % max(batch_size * 5, 20) < batch_size:
                rate = done / max(1e-6, time.time() - t0)
                log.info("Generated %d/%d (%.2f items/s)", done, len(prompts), rate)

        log.info("Batch generation complete: %d prompts in %.1fs.", len(prompts), time.time() - t0)
        return outputs

    def close(self) -> None:
        try:
            import torch

            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
