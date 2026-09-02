"""Prompt construction and output parsing.

The instruction text is taken verbatim from Ozsoy et al. (2025), Table 3, so
our zero-shot baselines are directly comparable to the published numbers for
the same dataset.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.logging_utils import get_logger

log = get_logger(__name__)

SYSTEM_INSTRUCTION = (
    "Task: Generate Cypher statement to query a graph database.\n"
    "Instructions: Use only the provided relationship types and properties in the schema. "
    "Do not use any other relationship types or properties that are not provided in the schema. "
    "Do not include any explanations or apologies in your responses. "
    "Do not respond to any questions that might ask anything else than for you to construct "
    "a Cypher statement. Do not include any text except the generated Cypher statement."
)

USER_TEMPLATE = (
    "Generate Cypher statement to query a graph database.\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n"
    "Cypher output:"
)


def build_messages(question: str, schema: str,
                   supports_system_role: bool = True) -> List[Dict[str, str]]:
    """Build a chat-format prompt.

    Gemma-family templates reject a `system` role, so the instruction is folded
    into the leading user turn instead. The token content is equivalent.
    """
    user = USER_TEMPLATE.format(schema=schema or "", question=question or "")
    if supports_system_role:
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user},
        ]
    return [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{user}"}]


def template_supports_system(tokenizer) -> bool:
    """Probe whether the tokenizer's chat template accepts a system role."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def template_supports_thinking(tokenizer) -> bool:
    src = getattr(tokenizer, "chat_template", None)
    if not isinstance(src, str):
        return False
    return "enable_thinking" in src


def render_prompt(
    tokenizer,
    question: str,
    schema: str,
    supports_system_role: Optional[bool] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the final prompt string, including the generation prefix.

    Both the HF and llama.cpp backends call this, guaranteeing byte-identical
    prompts across backends -- otherwise the laptop-vs-Colab comparison is
    meaningless.
    """
    if supports_system_role is None:
        supports_system_role = template_supports_system(tokenizer)

    kwargs = dict(chat_template_kwargs or {})
    # Only forward enable_thinking if the template actually reads it; passing it
    # to a template that does not is harmless but noisy, and some strict
    # templates raise on unexpected kwargs.
    if "enable_thinking" in kwargs and not template_supports_thinking(tokenizer):
        kwargs.pop("enable_thinking")

    messages = build_messages(question, schema, supports_system_role)

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception as exc:
            log.debug("apply_chat_template failed (%s); retrying without kwargs.", exc)
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as exc2:
                log.warning("Chat template unusable (%s); falling back to plain text.", exc2)

    # Plain-text fallback for base models with no chat template.
    body = "\n\n".join(m["content"] for m in messages)
    return f"{body}\n"


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:cypher|cql|sql|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_LEADIN = re.compile(r"^\s*(cypher\s*(query|statement|output)?\s*:|output\s*:|answer\s*:)\s*",
                     re.IGNORECASE)

_CYPHER_STARTERS = (
    "match", "merge", "create", "call", "with", "unwind", "return", "optional",
    "detach", "load", "use", "profile", "explain", "set", "delete", "remove", "foreach",
)

_STOP_MARKERS = ("<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<eos>", "<|eot_id|>")


def extract_cypher(raw: str) -> str:
    """Pull a single Cypher statement out of a model response.

    Handles reasoning blocks, code fences, lead-in phrases and trailing prose.
    """
    if not raw:
        return ""

    text = raw
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    # Reasoning traces.
    text = _THINK_BLOCK.sub(" ", text)
    if "</think>" in text.lower():
        text = _UNCLOSED_THINK.sub("", text, count=1)
    text = text.replace("<think>", " ")

    # Fenced code wins if present.
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1)

    text = _LEADIN.sub("", text.strip())

    # Keep contiguous blocks; drop trailing explanation paragraphs.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return ""

    chosen = blocks[0]
    for block in blocks:
        first_word = block.lstrip().split(" ", 1)[0].strip("`(").lower()
        if first_word in _CYPHER_STARTERS:
            chosen = block
            break

    # Strip stray leading backticks/quotes the model may emit.
    chosen = chosen.strip().strip("`").strip()
    # Drop a trailing semicolon so string comparison is not penalised by style.
    chosen = chosen.rstrip().rstrip(";").rstrip()
    return chosen
