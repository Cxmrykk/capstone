"""Translation-based metrics and query complexity classification.

Google-BLEU (GLEU) is implemented directly here rather than pulled from `evaluate`
to avoid runtime network dependencies on Colab sessions and ensure offline
evaluation capability.

GLEU follows Wu et al. (2016): for n = 1..4, take the total n-gram overlap
between hypothesis and reference, then take the minimum of
overlap/|hypothesis n-grams| and overlap/|reference n-grams|.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Code-aware tokenisation: identifiers, numbers, string literals, punctuation.
_TOKEN_RE = re.compile(
    r"""
    [A-Za-z_][A-Za-z_0-9]*      # identifiers / keywords
  | \d+\.\d+                    # floats
  | \d+                         # ints
  | "(?:[^"\\]|\\.)*"           # double-quoted literals
  | '(?:[^'\\]|\\.)*'           # single-quoted literals
  | [^\sA-Za-z0-9_]             # punctuation, arrows, braces
    """,
    re.VERBOSE,
)


def tokenize_cypher(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def _ngram_counts(tokens: Sequence[str], min_n: int = 1, max_n: int = 4) -> Counter:
    counts: Counter = Counter()
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i:i + n])] += 1
    return counts


def sentence_gleu(reference: str, hypothesis: str,
                  min_n: int = 1, max_n: int = 4) -> float:
    ref_tokens = tokenize_cypher(reference)
    hyp_tokens = tokenize_cypher(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts = _ngram_counts(ref_tokens, min_n, max_n)
    hyp_counts = _ngram_counts(hyp_tokens, min_n, max_n)
    overlap = sum((ref_counts & hyp_counts).values())

    n_hyp = sum(hyp_counts.values())
    n_ref = sum(ref_counts.values())
    if n_hyp == 0 or n_ref == 0:
        return 0.0
    return min(overlap / n_hyp, overlap / n_ref)


def corpus_gleu(references: Sequence[str], hypotheses: Sequence[str],
                min_n: int = 1, max_n: int = 4) -> float:
    """Corpus-level GLEU: aggregate counts first, then take the min of the ratios."""
    total_overlap = total_hyp = total_ref = 0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize_cypher(ref)
        hyp_tokens = tokenize_cypher(hyp)
        ref_counts = _ngram_counts(ref_tokens, min_n, max_n)
        hyp_counts = _ngram_counts(hyp_tokens, min_n, max_n)
        total_overlap += sum((ref_counts & hyp_counts).values())
        total_hyp += sum(hyp_counts.values())
        total_ref += sum(ref_counts.values())
    if total_hyp == 0 or total_ref == 0:
        return 0.0
    return min(total_overlap / total_hyp, total_overlap / total_ref)


# --------------------------------------------------------------------------- #
# Exact match
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_CYPHER_KEYWORDS = {
    "match", "optional", "where", "return", "with", "order", "by", "skip", "limit",
    "create", "merge", "delete", "detach", "set", "remove", "unwind", "call",
    "yield", "union", "all", "distinct", "as", "and", "or", "not", "in", "is",
    "null", "true", "false", "asc", "desc", "case", "when", "then", "else", "end",
    "contains", "starts", "ends", "count", "sum", "avg", "min", "max", "collect",
    "exists", "xor", "on", "foreach", "using",
}


def normalise_cypher(text: str) -> str:
    """Whitespace-insensitive, keyword-case-insensitive normalisation."""
    if not text:
        return ""
    tokens = tokenize_cypher(text)
    out: List[str] = []
    for tok in tokens:
        if tok.lower() in _CYPHER_KEYWORDS:
            out.append(tok.upper())
        else:
            out.append(tok)
    joined = " ".join(out)
    joined = _WS.sub(" ", joined).strip()
    return joined.rstrip(";").strip()


def exact_match(reference: str, hypothesis: str) -> bool:
    return (reference or "").strip().rstrip(";").strip() == \
           (hypothesis or "").strip().rstrip(";").strip()


def normalised_exact_match(reference: str, hypothesis: str) -> bool:
    return normalise_cypher(reference) == normalise_cypher(hypothesis)


# --------------------------------------------------------------------------- #
# Query complexity (Lyu et al. 2026, section 4.5)
# --------------------------------------------------------------------------- #
_AGG_RE = re.compile(r"\b(count|sum|avg|min|max|collect|percentile\w*|stdev\w*)\s*\(",
                     re.IGNORECASE)
_VARLEN_RE = re.compile(r"\[\s*:?\w*\s*\*")
_MATCH_RE = re.compile(r"\bmatch\b", re.IGNORECASE)
_WITH_RE = re.compile(r"\bwith\b", re.IGNORECASE)
_REL_RE = re.compile(r"-\s*\[")


@dataclass
class Complexity:
    label: str
    hops: int
    match_clauses: int
    aggregations: int
    variable_length: bool
    with_clauses: int


def classify_complexity(cypher: str) -> Complexity:
    """Heuristic difficulty label to segment results across complexity tiers
    (easy / medium / hard / extra_hard)."""
    text = cypher or ""
    hops = len(_REL_RE.findall(text))
    matches = len(_MATCH_RE.findall(text))
    aggs = len(_AGG_RE.findall(text))
    varlen = bool(_VARLEN_RE.search(text))
    withs = len(_WITH_RE.findall(text))

    if varlen or hops >= 4 or matches >= 3 or (withs >= 2 and aggs >= 1):
        label = "extra_hard"
    elif hops >= 2 or matches >= 2 or (aggs >= 1 and hops >= 1):
        label = "hard"
    elif hops == 1 or aggs >= 1 or withs >= 1:
        label = "medium"
    else:
        label = "easy"

    return Complexity(label, hops, matches, aggs, varlen, withs)


COMPLEXITY_ORDER = ["easy", "medium", "hard", "extra_hard"]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarise(rows: List[Dict], key_prefix: str = "") -> Dict[str, float]:
    """Aggregate per-row metric dicts into corpus-level numbers."""
    if not rows:
        return {}

    refs = [r["gold_cypher"] for r in rows]
    hyps = [r["predicted_cypher"] for r in rows]

    out: Dict[str, float] = {
        f"{key_prefix}n": len(rows),
        f"{key_prefix}google_bleu_corpus": round(corpus_gleu(refs, hyps), 4),
        f"{key_prefix}google_bleu_mean": round(
            sum(r.get("gleu", 0.0) for r in rows) / len(rows), 4),
        f"{key_prefix}exact_match": round(
            sum(1 for r in rows if r.get("exact_match")) / len(rows), 4),
        f"{key_prefix}normalised_exact_match": round(
            sum(1 for r in rows if r.get("normalised_exact_match")) / len(rows), 4),
        f"{key_prefix}empty_output_rate": round(
            sum(1 for r in rows if not r["predicted_cypher"].strip()) / len(rows), 4),
    }

    executed = [r for r in rows if r.get("execution_status") is not None]
    if executed:
        out[f"{key_prefix}n_executed"] = len(executed)
        out[f"{key_prefix}executable_rate"] = round(
            sum(1 for r in executed if r.get("pred_executed_ok")) / len(executed), 4)
        out[f"{key_prefix}execution_accuracy"] = round(
            sum(1 for r in executed if r.get("execution_match")) / len(executed), 4)

    validated = [r for r in rows if r.get("syntax_valid") is not None]
    if validated:
        out[f"{key_prefix}syntax_valid_rate"] = round(
            sum(1 for r in validated if r.get("syntax_valid")) / len(validated), 4)

    tok = [r.get("prompt_tokens") for r in rows
           if isinstance(r.get("prompt_tokens"), int) and r["prompt_tokens"] >= 0]
    if tok:
        tok_sorted = sorted(tok)
        out[f"{key_prefix}prompt_tokens_mean"] = round(sum(tok) / len(tok), 1)
        out[f"{key_prefix}prompt_tokens_p95"] = tok_sorted[
            min(len(tok_sorted) - 1, int(len(tok_sorted) * 0.95))]

    return out


def score_row(row: Dict) -> Dict:
    """Attach translation metrics and a complexity label to a prediction row."""
    gold = row.get("gold_cypher") or ""
    pred = row.get("predicted_cypher") or ""
    row["gleu"] = round(sentence_gleu(gold, pred), 4)
    row["exact_match"] = exact_match(gold, pred)
    row["normalised_exact_match"] = normalised_exact_match(gold, pred)
    complexity = classify_complexity(gold)
    row["complexity"] = complexity.label
    row["gold_hops"] = complexity.hops
    return row
