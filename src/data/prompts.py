"""Prompt formatting, chat template rendering, and Cypher statement extraction.

Constructs consistent instruction prompts for Text2Cypher tasks across different
model tokenizer formats and provides robust regex extractors for generated outputs.
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
    """Builds a structured list of chat messages.

    For model families whose chat templates reject a distinct system turn (such
    as Gemma), the system instruction is prepended to the initial user prompt.
    """
    user = USER_TEMPLATE.format(schema=schema or "", question=question or "")
    if supports_system_role:
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user},
        ]
    return [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{user}"}]


def template_supports_system(tokenizer) -> bool:
    """Probes whether the tokenizer chat template accepts a system role."""
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
    """Checks whether the Jinja chat template supports reasoning control flags."""
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
    """Renders the final formatted prompt string with the generation prefix.

    Ensures identical prompt construction across both PyTorch and GGUF inference engines.
    """
    if supports_system_role is None:
        supports_system_role = template_supports_system(tokenizer)

    kwargs = dict(chat_template_kwargs or {})
    if "enable_thinking" in kwargs and not template_supports_thinking(tokenizer):
        kwargs.pop("enable_thinking")

    messages = build_messages(question, schema, supports_system_role)

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception as exc:
            log.debug("apply_chat_template failed with kwargs (%s); retrying without kwargs.", exc)
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as exc2:
                log.warning("Chat template unusable (%s); falling back to plain text formatting.", exc2)

    # Plain text fallback for base models lacking a dedicated chat template
    body = "\n\n".join(m["content"] for m in messages)
    return f"{body}\n"


# --------------------------------------------------------------------------- #
# Output Extraction
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
    """Extracts a clean Cypher query string from raw model text generation.

    Handles reasoning blocks (<think>), markdown code blocks, lead-in prefixes,
    and trailing explanatory sentences.
    """
    if not raw:
        return ""

    text = raw
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    # Strip thinking/reasoning blocks
    text = _THINK_BLOCK.sub(" ", text)
    if "</think>" in text.lower():
        text = _UNCLOSED_THINK.sub("", text, count=1)
    text = text.replace("<think>", " ")

    # Extract markdown code fence if present
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1)

    text = _LEADIN.sub("", text.strip())

    # Locate contiguous code block starting with valid Cypher keywords
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return ""

    chosen = blocks[0]
    for block in blocks:
        first_word = block.lstrip().split(" ", 1)[0].strip("`(").lower()
        if first_word in _CYPHER_STARTERS:
            chosen = block
            break

    # Strip stray quotes, backticks, and trailing semicolons for consistency
    chosen = chosen.strip().strip("`").strip()
    chosen = chosen.rstrip().rstrip(";").rstrip()
    return chosen
