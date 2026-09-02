"""Schema representation and filtering (Ozsoy 2025, section 3).

Two schema string dialects appear in text2cypher-2024v1:

  A) markdown-ish
       Node properties:
       - **Country**
         - `code`: STRING Example: "AFG"
       The relationships:
       (:Filing)-[:BENEFITS]->(:Entity)

  B) inline
       Node properties are the following:
       Movie {title: STRING, released: INTEGER}, Person {name: STRING}
       The relationships are the following:
       (:Person)-[:ACTED_IN]->(:Movie)

The parser handles both and, crucially, degrades gracefully: if parsing fails
we return the schema untouched rather than silently feeding the model a
mangled context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.logging_utils import get_logger

log = get_logger(__name__)

# ``Example: "AFG"`` / ``Min: 1, Max: 174`` suffixes -- the difference between
# Ozsoy's "Enhanced" and "Base" static schemas.
_EXAMPLE_SUFFIX = re.compile(r"\s*(Example:.*|Min:.*?Max:.*?)$", re.IGNORECASE)

_PATTERN_RE = re.compile(r"\(:?([A-Za-z_][\w]*)?\)\s*-\s*\[:([A-Za-z_][\w]*)\]\s*->\s*\(:?([A-Za-z_][\w]*)?\)")
_MD_LABEL_RE = re.compile(r"^\s*-\s*\*\*([^*]+)\*\*\s*$")
_MD_PROP_RE = re.compile(r"^\s*-\s*`([^`]+)`\s*:\s*(.*)$")
_INLINE_ENTITY_RE = re.compile(r"([A-Za-z_][\w]*)\s*\{([^}]*)\}")

_SECTION_NODE = re.compile(r"^\s*node properties", re.IGNORECASE)
_SECTION_REL = re.compile(r"^\s*relationship properties", re.IGNORECASE)
_SECTION_PATTERNS = re.compile(r"^\s*the relationships", re.IGNORECASE)


@dataclass
class GraphSchema:
    nodes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    rel_props: Dict[str, Dict[str, str]] = field(default_factory=dict)
    patterns: List[Tuple[str, str, str]] = field(default_factory=list)
    raw: str = ""
    parsed: bool = False

    def is_empty(self) -> bool:
        return not self.nodes and not self.rel_props and not self.patterns

    def render(self, include_examples: bool = True) -> str:
        """Serialise back to the canonical markdown-ish dialect."""
        if not self.parsed:
            return self.raw

        out: List[str] = ["Node properties:"]
        for label, props in self.nodes.items():
            out.append(f"- **{label}**")
            for name, desc in props.items():
                desc_txt = desc if include_examples else _EXAMPLE_SUFFIX.sub("", desc).strip()
                out.append(f"  - `{name}`: {desc_txt}".rstrip())

        out.append("Relationship properties:")
        for rel, props in self.rel_props.items():
            out.append(f"- **{rel}**")
            for name, desc in props.items():
                desc_txt = desc if include_examples else _EXAMPLE_SUFFIX.sub("", desc).strip()
                out.append(f"  - `{name}`: {desc_txt}".rstrip())

        out.append("The relationships:")
        for src, rel, dst in self.patterns:
            out.append(f"(:{src})-[:{rel}]->(:{dst})")

        return "\n".join(out)

    def size(self) -> Dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "node_properties": sum(len(p) for p in self.nodes.values()),
            "relationship_types": len(self.rel_props),
            "patterns": len(self.patterns),
        }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_schema(raw: str) -> GraphSchema:
    schema = GraphSchema(raw=raw or "")
    if not raw or not raw.strip():
        return schema

    section = None
    current_label: Optional[str] = None
    current_kind: Optional[str] = None
    found_structure = False

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if _SECTION_NODE.match(stripped):
            section, current_label, current_kind = "node", None, None
            _absorb_inline(stripped, schema, "node")
            continue
        if _SECTION_REL.match(stripped):
            section, current_label, current_kind = "rel", None, None
            _absorb_inline(stripped, schema, "rel")
            continue
        if _SECTION_PATTERNS.match(stripped):
            section, current_label, current_kind = "patterns", None, None
            for m in _PATTERN_RE.finditer(stripped):
                schema.patterns.append((m.group(1) or "", m.group(2), m.group(3) or ""))
                found_structure = True
            continue

        # Relationship patterns can appear anywhere.
        pattern_hits = list(_PATTERN_RE.finditer(stripped))
        if pattern_hits:
            for m in pattern_hits:
                triple = (m.group(1) or "", m.group(2), m.group(3) or "")
                if triple not in schema.patterns:
                    schema.patterns.append(triple)
                    found_structure = True
            continue

        label_match = _MD_LABEL_RE.match(line)
        if label_match and section in {"node", "rel"}:
            current_label = label_match.group(1).strip()
            current_kind = section
            target = schema.nodes if section == "node" else schema.rel_props
            target.setdefault(current_label, {})
            found_structure = True
            continue

        prop_match = _MD_PROP_RE.match(line)
        if prop_match and current_label and current_kind:
            target = schema.nodes if current_kind == "node" else schema.rel_props
            target.setdefault(current_label, {})[prop_match.group(1).strip()] = \
                prop_match.group(2).strip()
            found_structure = True
            continue

        # Dialect B: "Movie {title: STRING}, Person {name: STRING}"
        if section in {"node", "rel"} and "{" in stripped:
            _absorb_inline(stripped, schema, section)
            found_structure = True
            continue

    schema.parsed = found_structure
    if not found_structure:
        log.debug("Schema parsing found no structure; will pass through unchanged.")
    return schema


def _absorb_inline(line: str, schema: GraphSchema, kind: str) -> None:
    target = schema.nodes if kind == "node" else schema.rel_props
    for m in _INLINE_ENTITY_RE.finditer(line):
        label = m.group(1).strip()
        body = m.group(2)
        props: Dict[str, str] = {}
        for part in body.split(","):
            if ":" not in part:
                continue
            name, _, typ = part.partition(":")
            props[name.strip()] = typ.strip()
        if label:
            target.setdefault(label, {}).update(props)


# --------------------------------------------------------------------------- #
# Tokenisation helpers
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Deliberately small: only words that would cause spurious schema matches.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with", "by",
    "what", "which", "who", "whom", "how", "many", "much", "list", "show", "give",
    "find", "get", "all", "are", "is", "was", "were", "be", "have", "has", "had",
    "return", "count", "most", "least", "top", "there", "their", "that", "this",
    "from", "as", "at", "it", "do", "does", "did", "me", "please", "name", "names",
}


def _explode(term: str) -> Set[str]:
    """Turn `originator_bank_country` / `ACTED_IN` into searchable word pieces."""
    pieces: Set[str] = set()
    term = term.strip()
    if not term:
        return pieces
    pieces.add(term.lower())
    for chunk in re.split(r"[_\-\s.]+", term):
        if not chunk:
            continue
        pieces.add(chunk.lower())
        for sub in _CAMEL_RE.split(chunk):
            if len(sub) > 1:
                pieces.add(sub.lower())
    return {p for p in pieces if len(p) > 1}


def question_tokens(question: str) -> Set[str]:
    toks: Set[str] = set()
    for word in _WORD_RE.findall(question or ""):
        low = word.lower()
        if low in _STOPWORDS:
            continue
        toks |= _explode(word)
        # Crude singularisation so "movies" matches the `Movie` label.
        if low.endswith("ies") and len(low) > 4:
            toks.add(low[:-3] + "y")
        elif low.endswith("ses") and len(low) > 4:
            toks.add(low[:-2])
        elif low.endswith("s") and not low.endswith("ss") and len(low) > 3:
            toks.add(low[:-1])
    return toks


_QUOTED = re.compile(r"(\"[^\"]*\"|'[^']*')")
_TITLECASE_SPAN = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+(?:[A-Z][a-z]+|of|the|and|de|van))*\b")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")


def ner_mask(question: str) -> str:
    """Heuristic stand-in for NER masking (Ozsoy 2025, section 3.2).

    Real NER needs spaCy or an NER model; the goal here is only to stop literal
    entity values ("Tom Hanks", "United Kingdom") from matching schema labels.
    Documented as a heuristic in the write-up.
    """
    if not question:
        return ""
    masked = _QUOTED.sub(" ENTITY ", question)
    masked = _NUMBER.sub(" NUMBER ", masked)

    def _sub(m: re.Match) -> str:
        span = m.group(0)
        # Do not mask the very first word of the sentence.
        if m.start() == 0:
            return span
        return " ENTITY " if " " in span or len(span) > 3 else span

    return _TITLECASE_SPAN.sub(_sub, masked)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def filter_exact_match(
    schema: GraphSchema,
    question: str,
    min_nodes: int = 1,
    keep_patterns_for_kept_nodes: bool = True,
) -> GraphSchema:
    if not schema.parsed:
        return schema

    tokens = question_tokens(question)
    if not tokens:
        return schema

    kept_nodes: Dict[str, Dict[str, str]] = {}
    for label, props in schema.nodes.items():
        label_hit = bool(_explode(label) & tokens)
        matched_props = {n: d for n, d in props.items() if _explode(n) & tokens}
        if label_hit:
            # Label matched: keep the full property list for that node.
            kept_nodes[label] = dict(props)
        elif matched_props:
            kept_nodes[label] = matched_props

    kept_rels: Dict[str, Dict[str, str]] = {}
    for rel, props in schema.rel_props.items():
        if _explode(rel) & tokens:
            kept_rels[rel] = dict(props)
        else:
            matched = {n: d for n, d in props.items() if _explode(n) & tokens}
            if matched:
                kept_rels[rel] = matched

    kept_patterns: List[Tuple[str, str, str]] = []
    for src, rel, dst in schema.patterns:
        rel_hit = bool(_explode(rel) & tokens)
        endpoints_kept = src in kept_nodes and dst in kept_nodes
        if rel_hit or (keep_patterns_for_kept_nodes and endpoints_kept):
            kept_patterns.append((src, rel, dst))

    # Re-attach any node referenced by a surviving pattern; a pattern with an
    # undefined endpoint is worse than no pruning at all.
    for src, _rel, dst in kept_patterns:
        for endpoint in (src, dst):
            if endpoint and endpoint not in kept_nodes and endpoint in schema.nodes:
                kept_nodes[endpoint] = schema.nodes[endpoint]

    if len(kept_nodes) < min_nodes and not kept_patterns:
        # Over-pruned: an empty schema guarantees a wrong query, so fall back.
        return schema

    return GraphSchema(
        nodes=kept_nodes,
        rel_props=kept_rels,
        patterns=kept_patterns,
        raw=schema.raw,
        parsed=True,
    )


def filter_by_similarity(
    schema: GraphSchema,
    question: str,
    threshold: float = 0.45,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    _cache: Dict[str, object] = {},
) -> GraphSchema:
    """Embedding-based pruning. Optional dependency; degrades to exact match."""
    if not schema.parsed:
        return schema
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        log.warning("sentence-transformers not installed; falling back to exact-match pruning.")
        return filter_exact_match(schema, question)

    model = _cache.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _cache[model_name] = model

    elements: List[Tuple[str, str, str]] = []   # (kind, owner, name)
    texts: List[str] = []
    for label, props in schema.nodes.items():
        elements.append(("node", label, label))
        texts.append(label)
        for name in props:
            elements.append(("node_prop", label, name))
            texts.append(f"{label} {name}")
    for rel, props in schema.rel_props.items():
        elements.append(("rel", rel, rel))
        texts.append(rel)
        for name in props:
            elements.append(("rel_prop", rel, name))
            texts.append(f"{rel} {name}")

    if not texts:
        return schema

    q_emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
    e_emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, e_emb)[0].tolist()

    kept_nodes: Dict[str, Dict[str, str]] = {}
    kept_rels: Dict[str, Dict[str, str]] = {}
    for (kind, owner, name), score in zip(elements, scores):
        if score < threshold:
            continue
        if kind == "node":
            kept_nodes.setdefault(owner, dict(schema.nodes[owner]))
        elif kind == "node_prop":
            kept_nodes.setdefault(owner, {})[name] = schema.nodes[owner][name]
        elif kind == "rel":
            kept_rels.setdefault(owner, dict(schema.rel_props[owner]))
        elif kind == "rel_prop":
            kept_rels.setdefault(owner, {})[name] = schema.rel_props[owner][name]

    kept_patterns = [
        (s, r, d) for (s, r, d) in schema.patterns
        if r in kept_rels or (s in kept_nodes and d in kept_nodes)
    ]
    for src, _rel, dst in kept_patterns:
        for endpoint in (src, dst):
            if endpoint and endpoint not in kept_nodes and endpoint in schema.nodes:
                kept_nodes[endpoint] = schema.nodes[endpoint]

    if not kept_nodes and not kept_patterns:
        return schema

    return GraphSchema(nodes=kept_nodes, rel_props=kept_rels,
                       patterns=kept_patterns, raw=schema.raw, parsed=True)


def apply_schema_mode(
    raw_schema: str,
    question: str,
    mode: str,
    min_nodes: int = 1,
    keep_patterns_for_kept_nodes: bool = True,
    similarity_threshold: float = 0.45,
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> str:
    """Single entry point used by the dataset builder and the predictor."""
    if mode == "none":
        return ""
    if mode == "enhanced":
        return raw_schema or ""

    schema = parse_schema(raw_schema)
    if not schema.parsed:
        # Unparseable dialect: better to keep the original than to guess.
        return raw_schema or ""

    if mode == "base":
        return schema.render(include_examples=False)

    if mode == "exact_match":
        pruned = filter_exact_match(schema, question, min_nodes,
                                    keep_patterns_for_kept_nodes)
        return pruned.render(include_examples=True)

    if mode == "ner_exact_match":
        pruned = filter_exact_match(schema, ner_mask(question), min_nodes,
                                    keep_patterns_for_kept_nodes)
        return pruned.render(include_examples=True)

    if mode == "similarity":
        pruned = filter_by_similarity(schema, question, similarity_threshold,
                                      similarity_model)
        return pruned.render(include_examples=True)

    raise ValueError(f"Unknown schema mode: {mode}")


def trim_schema_text(schema_text: str, drop_chars: int) -> str:
    """Drop trailing property lines to fit a token budget.

    Removes leaf property lines first and never removes the relationship
    pattern block, which carries most of the structural signal.
    """
    if drop_chars <= 0 or not schema_text:
        return schema_text

    lines = schema_text.splitlines()
    protected_from = len(lines)
    for i, line in enumerate(lines):
        if _SECTION_PATTERNS.match(line.strip()):
            protected_from = i
            break

    removable = [i for i in range(protected_from)
                 if _MD_PROP_RE.match(lines[i])]
    removed = 0
    to_drop: Set[int] = set()
    for idx in reversed(removable):
        if removed >= drop_chars:
            break
        removed += len(lines[idx]) + 1
        to_drop.add(idx)

    kept = [ln for i, ln in enumerate(lines) if i not in to_drop]
    if to_drop:
        kept.insert(min(protected_from, len(kept)), "  ... (properties truncated)")
    return "\n".join(kept)
