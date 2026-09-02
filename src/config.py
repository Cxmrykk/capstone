"""Configuration management and base model registry.

Encapsulates all runtime, training, data processing, generation, and evaluation
parameters to ensure reproducible runs via YAML files and CLI overrides.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logging_utils import get_logger
from src.paths import data_dir, repo_root, runs_dir

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Model Registry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """Metadata and architectural specifications for supported base models."""

    key: str
    hf_id: str
    local_dirname: str
    family: str                      # "gemma" | "qwen" | "generic"
    attn_implementation: str = "sdpa"
    # Arguments passed to tokenizer.apply_chat_template(). For instance, disabling
    # reasoning traces prevents unwanted prefix tokens during instruction tuning.
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    supports_system_role: Optional[bool] = None   # None triggers runtime probing
    # Module path substrings that should be excluded from LoRA adaptation
    lora_exclude_substrings: List[str] = field(default_factory=lambda: [
        "vision", "visual", "audio", "mm_projector", "multi_modal",
        "embed_tokens", "lm_head", "per_layer", "altup", "conv1d",
    ])
    notes: str = ""

    def resolve_path(self, override: Optional[str] = None) -> str:
        """Resolves model path, prioritizing local checkouts over remote Hub identifiers."""
        if override:
            return str(Path(override).expanduser())
        env = os.environ.get(f"T2C_MODEL_{self.key.upper().replace('.', '_').replace('-', '_')}")
        if env:
            return env
        local = data_dir() / self.local_dirname
        if local.exists() and any(local.iterdir()):
            return str(local)
        return self.hf_id


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gemma-4-e2b": ModelSpec(
        key="gemma-4-e2b",
        hf_id="google/gemma-4-E2B",
        local_dirname="gemma-4-E2B",
        family="gemma",
        attn_implementation="eager",
        supports_system_role=False,
        notes="Multimodal architecture; adapts the core language transformer layers.",
    ),
    "gemma-4-e4b": ModelSpec(
        key="gemma-4-e4b",
        hf_id="google/gemma-4-E4B",
        local_dirname="gemma-4-E4B",
        family="gemma",
        attn_implementation="eager",
        supports_system_role=False,
        notes="Multimodal architecture; adapts the core language transformer layers.",
    ),
    "qwen3.5-2b": ModelSpec(
        key="qwen3.5-2b",
        hf_id="Qwen/Qwen3.5-2B",
        local_dirname="Qwen3.5-2B",
        family="qwen",
        attn_implementation="sdpa",
        chat_template_kwargs={"enable_thinking": False},
        supports_system_role=True,
        notes="Hybrid attention model with standard chat template support.",
    ),
    "qwen3.5-4b": ModelSpec(
        key="qwen3.5-4b",
        hf_id="Qwen/Qwen3.5-4B",
        local_dirname="Qwen3.5-4B",
        family="qwen",
        attn_implementation="sdpa",
        chat_template_kwargs={"enable_thinking": False},
        supports_system_role=True,
        notes="Hybrid attention model with standard chat template support.",
    ),
}

# CLI aliases for model keys
MODEL_ALIASES = {
    "gemma-2b": "gemma-4-e2b",
    "gemma2b": "gemma-4-e2b",
    "gemma-e2b": "gemma-4-e2b",
    "gemma-4b": "gemma-4-e4b",
    "gemma4b": "gemma-4-e4b",
    "gemma-e4b": "gemma-4-e4b",
    "qwen-2b": "qwen3.5-2b",
    "qwen2b": "qwen3.5-2b",
    "qwen-4b": "qwen3.5-4b",
    "qwen4b": "qwen3.5-4b",
}


def get_model_spec(key: str) -> ModelSpec:
    norm = key.strip().lower()
    norm = MODEL_ALIASES.get(norm, norm)
    if norm not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY) + sorted(MODEL_ALIASES))
        raise KeyError(f"Unknown model key '{key}'. Supported models: {known}")
    return MODEL_REGISTRY[norm]


# --------------------------------------------------------------------------- #
# Config Sections
# --------------------------------------------------------------------------- #
SCHEMA_MODES = ("enhanced", "base", "exact_match", "ner_exact_match", "similarity", "none")


@dataclass
class ModelConfig:
    key: str = "qwen3.5-2b"
    path: Optional[str] = None
    load_in_4bit: bool = True
    attn_implementation: str = "auto"     # "auto" uses the ModelSpec default
    use_unsloth: str = "auto"             # "auto" | "yes" | "no"
    trust_remote_code: bool = True

    @property
    def spec(self) -> ModelSpec:
        return get_model_spec(self.key)

    def resolved_path(self) -> str:
        return self.spec.resolve_path(self.path)

    def resolved_attn(self) -> str:
        return self.spec.attn_implementation if self.attn_implementation == "auto" \
            else self.attn_implementation


@dataclass
class DataConfig:
    dataset_dir: Optional[str] = None
    schema_mode: str = "enhanced"
    max_seq_length: int = 2048
    val_fraction: float = 0.02
    max_val_samples: int = 400
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    seed: int = 3407
    # Schema filtering parameters
    filter_min_nodes: int = 1
    filter_keep_all_patterns_for_kept_nodes: bool = True
    similarity_threshold: float = 0.45
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if self.schema_mode not in SCHEMA_MODES:
            raise ValueError(f"schema_mode must be one of {SCHEMA_MODES}, got {self.schema_mode!r}")


@dataclass
class LoraConfig_:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "none"
    target_modules: Any = "auto"       # "auto" | "all-linear" | list[str]
    modules_to_save: Optional[List[str]] = None


@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 250
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    group_by_length: bool = False
    dataloader_num_workers: int = 2
    seed: int = 3407
    time_limit_minutes: Optional[float] = None
    report_to: List[str] = field(default_factory=list)


@dataclass
class HubConfig:
    repo_id: Optional[str] = None
    private: bool = True
    sync: bool = True
    keep_last: int = 2
    token_env: str = "HF_TOKEN"
    async_upload: bool = True
    drive_mirror: Optional[str] = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 8
    do_sample: bool = False
    repetition_penalty: float = 1.0
    stop_strings: List[str] = field(default_factory=lambda: ["\n\n\n", "<|im_end|>", "<end_of_turn>"])


@dataclass
class Neo4jConfig:
    provider: str = "demo"                 # "demo" | "local"
    demo_uri: str = "neo4j+s://demo.neo4jlabs.com:7687"
    local_uri: str = "bolt://localhost:7687"
    local_user: str = "neo4j"
    local_password_env: str = "NEO4J_PASSWORD"
    local_database: str = "neo4j"
    local_alias: Optional[str] = None
    query_timeout_s: float = 30.0
    connection_timeout_s: float = 20.0
    min_interval_s: float = 0.05           # Rate-limiting pause between remote queries
    max_retries: int = 2
    order_sensitive: bool = True           # Maintain row order when ORDER BY is in gold query
    compare_keys: bool = False             # Validate returned column names in addition to values
    float_ndigits: int = 6


@dataclass
class RunConfig:
    run_name: str = "default"
    output_dir: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig_ = field(default_factory=LoraConfig_)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    config_path: Optional[str] = None

    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            path = Path(self.output_dir).expanduser()
        else:
            path = runs_dir() / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Generates a deterministic hash of training-critical parameters.

        Ensures that checkpoint resumption is refused if the underlying
        hyperparameters or dataset configurations have changed.
        """
        relevant = {
            "model": dataclasses.asdict(self.model),
            "data": dataclasses.asdict(self.data),
            "lora": dataclasses.asdict(self.lora),
            "train": {
                k: v for k, v in dataclasses.asdict(self.train).items()
                if k not in {"time_limit_minutes", "report_to", "save_steps",
                             "logging_steps", "dataloader_num_workers"}
            },
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload.pop("config_path", None)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# Loading Helpers
# --------------------------------------------------------------------------- #
_SECTION_TYPES = {
    "model": ModelConfig,
    "data": DataConfig,
    "lora": LoraConfig_,
    "train": TrainConfig,
    "hub": HubConfig,
    "generation": GenerationConfig,
    "neo4j": Neo4jConfig,
}


def _build_section(cls, payload: Dict[str, Any], name: str):
    valid = {f.name for f in dataclasses.fields(cls)}
    clean, unknown = {}, []
    for k, v in (payload or {}).items():
        if k in valid:
            clean[k] = v
        else:
            unknown.append(k)
    if unknown:
        log.warning("Ignoring unrecognized parameters in config section '%s': %s", name, ", ".join(unknown))
    return cls(**clean)


def config_from_dict(payload: Dict[str, Any]) -> RunConfig:
    payload = copy.deepcopy(payload or {})
    sections = {}
    for name, cls in _SECTION_TYPES.items():
        sections[name] = _build_section(cls, payload.pop(name, {}) or {}, name)

    run_name = payload.pop("run_name", "default")
    output_dir = payload.pop("output_dir", None)
    if payload:
        log.warning("Ignoring unrecognized top-level config keys: %s", ", ".join(payload))

    return RunConfig(run_name=run_name, output_dir=output_dir, **sections)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).expanduser()
    if not path.exists():
        candidate = repo_root() / "configs" / f"{path.name}.yaml"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)

    cfg = config_from_dict(payload)
    cfg.config_path = str(path)
    if cfg.run_name == "default":
        cfg.run_name = path.stem
    log.info("Loaded config '%s' (run_name=%s, fingerprint=%s)",
             path, cfg.run_name, cfg.fingerprint())
    return cfg
