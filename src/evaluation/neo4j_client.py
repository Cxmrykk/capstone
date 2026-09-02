"""Neo4j driver connectivity, query execution, and result verification.

Supported connection providers:
  * "demo"  -- Neo4j Labs public demo server. Resolves database aliases
               (e.g., neo4jlabs_demo_db_fincen) using demo endpoint conventions.
  * "local" -- Local Neo4j instance for offline development and testing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import Neo4jConfig
from src.logging_utils import get_logger
from src.paths import cache_dir

log = get_logger(__name__)

_ALIAS_PREFIXES = ("neo4jlabs_demo_db_", "neo4jlabs_demo_", "neo4j_demo_", "demo_db_")
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


@dataclass(frozen=True)
class Neo4jTarget:
    uri: str
    user: str
    password: str
    database: str
    alias: str

    def key(self) -> str:
        return f"{self.uri}|{self.database}"


def alias_to_database(alias: str) -> Optional[str]:
    if not alias:
        return None
    name = alias.strip()
    for prefix in _ALIAS_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.strip("_- ")
    return name or None


def resolve_target(alias: Optional[str], cfg: Neo4jConfig) -> Optional[Neo4jTarget]:
    """Resolves database alias to connection credentials based on the active provider."""
    db = alias_to_database(alias) if alias else None
    if not db:
        return None

    if cfg.provider == "demo":
        uri = os.environ.get("NEO4J_DEMO_URI", cfg.demo_uri)
        return Neo4jTarget(uri=uri, user=db, password=db, database=db, alias=alias or db)

    if cfg.provider == "local":
        if cfg.local_alias and alias_to_database(cfg.local_alias) != db:
            return None
        password = os.environ.get(cfg.local_password_env, "")
        if not password:
            log.warning("Environment variable %s is not set; local Neo4j authentication may fail.",
                        cfg.local_password_env)
        return Neo4jTarget(
            uri=os.environ.get("NEO4J_URI", cfg.local_uri),
            user=os.environ.get("NEO4J_USER", cfg.local_user),
            password=password,
            database=cfg.local_database,
            alias=alias or db,
        )

    raise ValueError(f"Unknown Neo4j provider: {cfg.provider!r}")


# --------------------------------------------------------------------------- #
# Result Normalization
# --------------------------------------------------------------------------- #
def _normalise_value(value: Any, ndigits: int) -> Any:
    """Reduces Neo4j driver types to standard JSON-compatible primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, (list, tuple)):
        return [_normalise_value(v, ndigits) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalise_value(v, ndigits) for k, v in sorted(value.items())}
    if isinstance(value, (bytes, bytearray)):
        return value.hex()

    try:
        from neo4j.graph import Node, Path, Relationship

        if isinstance(value, Node):
            return {
                "_labels": sorted(value.labels),
                **{k: _normalise_value(v, ndigits) for k, v in sorted(dict(value).items())},
            }
        if isinstance(value, Relationship):
            return {
                "_type": value.type,
                **{k: _normalise_value(v, ndigits) for k, v in sorted(dict(value).items())},
            }
        if isinstance(value, Path):
            return {
                "_nodes": [_normalise_value(n, ndigits) for n in value.nodes],
                "_rels": [_normalise_value(r, ndigits) for r in value.relationships],
            }
    except Exception:
        pass

    return str(value)


def normalise_records(records: Sequence[Any], ndigits: int = 6
                      ) -> Tuple[List[str], List[List[Any]]]:
    """Extracts column headers and normalized value rows from Neo4j driver results."""
    keys: List[str] = []
    rows: List[List[Any]] = []
    for rec in records:
        try:
            data = rec.data()
        except Exception:
            data = dict(rec) if isinstance(rec, dict) else {"value": rec}
        if not keys:
            keys = list(data.keys())
        rows.append([_normalise_value(data.get(k), ndigits) for k in keys])
    return keys, rows


def result_signature(keys: Sequence[str], rows: Sequence[Sequence[Any]],
                     order_sensitive: bool, compare_keys: bool) -> str:
    """Builds a canonical signature string for deterministic result set comparison."""
    serialised = [json.dumps(row, sort_keys=True, default=str) for row in rows]
    if not order_sensitive:
        serialised = sorted(serialised)
    body = "\n".join(serialised)
    if compare_keys:
        return json.dumps({"keys": list(keys), "rows": body}, sort_keys=True)
    return body


def query_is_ordered(cypher: str) -> bool:
    return bool(_ORDER_BY_RE.search(cypher or ""))


# --------------------------------------------------------------------------- #
# Executor & Cache
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionResult:
    ok: bool
    keys: List[str]
    rows: List[List[Any]]
    error: Optional[str] = None
    elapsed_s: float = 0.0
    from_cache: bool = False

    def signature(self, order_sensitive: bool, compare_keys: bool) -> Optional[str]:
        if not self.ok:
            return None
        return result_signature(self.keys, self.rows, order_sensitive, compare_keys)


class Neo4jExecutor:
    """Manages driver connection pools, query throttling, and execution result caching."""

    def __init__(self, cfg: Neo4jConfig, use_cache: bool = True) -> None:
        self.cfg = cfg
        self.use_cache = use_cache
        self._drivers: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._cache_root = cache_dir() / "neo4j" / cfg.provider
        if use_cache:
            self._cache_root.mkdir(parents=True, exist_ok=True)
        self._unreachable: Dict[str, str] = {}

    def _driver(self, target: Neo4jTarget):
        with self._lock:
            key = target.key()
            if key in self._drivers:
                return self._drivers[key]
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise ImportError(
                    "The neo4j driver is required for query execution.\n"
                    "Install with: pip install neo4j"
                ) from exc

            driver = GraphDatabase.driver(
                target.uri,
                auth=(target.user, target.password),
                connection_timeout=self.cfg.connection_timeout_s,
                max_connection_lifetime=600,
            )
            self._drivers[key] = driver
            return driver

    def ping(self, target: Neo4jTarget) -> Tuple[bool, str]:
        try:
            driver = self._driver(target)
            driver.verify_connectivity()
            with driver.session(database=target.database) as session:
                session.run("RETURN 1 AS ok").consume()
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        with self._lock:
            for driver in self._drivers.values():
                try:
                    driver.close()
                except Exception:
                    pass
            self._drivers.clear()

    def _cache_path(self, target: Neo4jTarget, query: str, kind: str) -> Path:
        digest = hashlib.sha256(f"{target.key()}|{kind}|{query}".encode("utf-8")).hexdigest()
        sub = self._cache_root / target.database / digest[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{digest}.json"

    def _read_cache(self, path: Path) -> Optional[Dict[str, Any]]:
        if not self.use_cache or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, path: Path, payload: Dict[str, Any]) -> None:
        if not self.use_cache:
            return
        try:
            path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception as exc:
            log.debug("Cache write failed for %s: %s", path, exc)

    def _throttle(self) -> None:
        if self.cfg.min_interval_s <= 0:
            return
        delta = time.time() - self._last_call
        if delta < self.cfg.min_interval_s:
            time.sleep(self.cfg.min_interval_s - delta)
        self._last_call = time.time()

    def validate_syntax(self, target: Neo4jTarget, query: str) -> Tuple[bool, Optional[str]]:
        """Validates query syntax and schema alignment using EXPLAIN without full execution."""
        if not (query or "").strip():
            return False, "empty query"

        cache_file = self._cache_path(target, query, "explain")
        cached = self._read_cache(cache_file)
        if cached is not None:
            return cached["ok"], cached.get("error")

        try:
            self._throttle()
            driver = self._driver(target)
            with driver.session(database=target.database) as session:
                session.run(f"EXPLAIN {query}").consume()
            self._write_cache(cache_file, {"ok": True, "error": None})
            return True, None
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._write_cache(cache_file, {"ok": False, "error": msg})
            return False, msg

    def execute(self, target: Neo4jTarget, query: str) -> ExecutionResult:
        """Executes a Cypher query against the target database."""
        if not (query or "").strip():
            return ExecutionResult(ok=False, keys=[], rows=[], error="empty query")

        if target.key() in self._unreachable:
            return ExecutionResult(ok=False, keys=[], rows=[],
                                   error=f"unreachable: {self._unreachable[target.key()]}")

        cache_file = self._cache_path(target, query, "run")
        cached = self._read_cache(cache_file)
        if cached is not None:
            return ExecutionResult(
                ok=cached["ok"],
                keys=cached.get("keys", []),
                rows=cached.get("rows", []),
                error=cached.get("error"),
                elapsed_s=cached.get("elapsed_s", 0.0),
                from_cache=True,
            )

        last_error = "unknown"
        for attempt in range(self.cfg.max_retries + 1):
            try:
                self._throttle()
                t0 = time.time()
                driver = self._driver(target)
                with driver.session(database=target.database) as session:
                    result = session.run(query, timeout=self.cfg.query_timeout_s)
                    records = list(result)
                keys, rows = normalise_records(records, self.cfg.float_ndigits)
                payload = {
                    "ok": True, "keys": keys, "rows": rows,
                    "error": None, "elapsed_s": round(time.time() - t0, 3),
                }
                self._write_cache(cache_file, payload)
                return ExecutionResult(True, keys, rows, None, payload["elapsed_s"])

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                lowered = last_error.lower()
                if any(tok in lowered for tok in
                       ("authentication", "unauthorized", "servicunavailable",
                        "cannot resolve", "unable to retrieve routing")):
                    if attempt == 0:
                        log.error("Unable to connect to %s (%s): %s",
                                  target.database, target.uri, last_error)
                    self._unreachable[target.key()] = last_error
                    break
                if attempt < self.cfg.max_retries:
                    time.sleep(1.0 + attempt)

        payload = {"ok": False, "keys": [], "rows": [], "error": last_error, "elapsed_s": 0.0}
        if "unreachable" not in last_error.lower():
            self._write_cache(cache_file, payload)
        return ExecutionResult(False, [], [], last_error)

    def compare(self, target: Neo4jTarget, gold_cypher: str,
                pred_cypher: str) -> Dict[str, Any]:
        """Executes reference and predicted queries and compares resulting records."""
        order_sensitive = self.cfg.order_sensitive and query_is_ordered(gold_cypher)

        gold = self.execute(target, gold_cypher)
        pred = self.execute(target, pred_cypher)

        gold_sig = gold.signature(order_sensitive, self.cfg.compare_keys)
        pred_sig = pred.signature(order_sensitive, self.cfg.compare_keys)

        if not gold.ok:
            status = "gold_failed"
            match = None
        elif not pred.ok:
            status = "pred_failed"
            match = False
        else:
            status = "compared"
            match = gold_sig == pred_sig

        return {
            "execution_status": status,
            "execution_match": match,
            "gold_executed_ok": gold.ok,
            "pred_executed_ok": pred.ok,
            "gold_error": gold.error,
            "pred_error": pred.error,
            "gold_row_count": len(gold.rows),
            "pred_row_count": len(pred.rows),
            "order_sensitive": order_sensitive,
            "gold_from_cache": gold.from_cache,
            "pred_from_cache": pred.from_cache,
        }
