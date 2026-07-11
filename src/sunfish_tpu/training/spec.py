"""Strict, dependency-free configuration contract for Sunfish training.

This module must stay stdlib-only.  ``sunfish-train --validate-only`` imports
it before distributed JAX initialization, and the same immutable object is
used to derive the run-identity digest checked on every worker.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Phase(enum.StrEnum):
    """Parameter-update phase; each value has a distinct optimizer mask."""

    SMOKE = "smoke"
    ROUTER = "router"
    LORA = "lora"
    FULL = "full"


class TimeSampling(enum.StrEnum):
    """Noise-time distribution used for each prefix-amortized draw."""

    PREFIX_STRATIFIED = "prefix_stratified"
    UNIFORM = "uniform"
    LOGIT_NORMAL = "logit_normal"


class CheckpointFormat(enum.StrEnum):
    """Supported initialization sources for a fresh run/workdir."""

    EXACT_TREE = "orbax-exact-tree"
    KAULDRON_PARAMS = "kauldron-params"


@dataclasses.dataclass(frozen=True)
class RunSpec:
    run_id: str
    phase: Phase
    workdir: str
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    num_experts: int = 32
    top_k_experts: int = 4
    vocab_size: int = 262_144
    hidden_size: int = 2_816
    num_layers: int = 30
    expert_hidden_size: int = 704
    dtype: str = "bfloat16"


@dataclasses.dataclass(frozen=True)
class DataSpec:
    directory: str
    manifest_sha256: str
    global_batch_size: int
    prompt_length: int
    canvas_size: int = 256
    num_canvases: int = 1
    pad_token: int = 0
    eos_token: int = 1
    shuffle: bool = True
    num_workers: int = 16
    verify_shard_hashes: bool = False

    @property
    def total_canvas_length(self) -> int:
        return self.canvas_size * self.num_canvases


@dataclasses.dataclass(frozen=True)
class ObjectiveSpec:
    noise_draws: int = 4
    time_sampling: TimeSampling = TimeSampling.PREFIX_STRATIFIED
    self_condition_probability: float = 0.5
    decoder_loss_weight: float = 1.0
    encoder_loss_weight: float = 1.0
    stop_gradient_from_denoiser_to_encoder: bool = False
    safety_epsilon: float = 1e-4
    logit_normal_mean: float = 0.0
    logit_normal_scale: float = 1.0


@dataclasses.dataclass(frozen=True)
class OptimizerSpec:
    peak_learning_rate: float
    end_learning_rate: float
    warmup_steps: int
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    adam_beta1: float = 0.95
    adam_beta2: float = 0.99
    adam_epsilon: float = 1e-8
    lora_rank: int = 16


@dataclasses.dataclass(frozen=True)
class TrainingSpec:
    steps: int
    checkpoint_every_steps: int
    max_checkpoints_to_keep: int = 3
    log_metrics_every_steps: int = 10
    log_summaries_every_steps: int = 100


@dataclasses.dataclass(frozen=True)
class CheckpointSpec:
    init_path: str
    format: CheckpointFormat = CheckpointFormat.EXACT_TREE
    model_param_path: str = "gemma_network.gemma_model"
    init_step: int = -1


@dataclasses.dataclass(frozen=True)
class TopologySpec:
    require_tpu: bool = True
    expected_devices: int = 0
    expected_processes: int = 0
    expected_local_devices: int = 0


@dataclasses.dataclass(frozen=True)
class HarnessConfig:
    """Complete immutable input to one training run."""

    schema_version: int
    run: RunSpec
    model: ModelSpec
    data: DataSpec
    objective: ObjectiveSpec
    optimizer: OptimizerSpec
    training: TrainingSpec
    checkpoint: CheckpointSpec
    topology: TopologySpec

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as source:
            payload = tomllib.load(source)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        _expect_keys(
            "root",
            payload,
            required={
                "schema_version",
                "run",
                "model",
                "data",
                "objective",
                "optimizer",
                "training",
                "checkpoint",
                "topology",
            },
        )
        config = cls(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            run=_build_dataclass(RunSpec, payload["run"], enums={"phase": Phase}),
            model=_build_dataclass(ModelSpec, payload["model"]),
            data=_build_dataclass(DataSpec, payload["data"]),
            objective=_build_dataclass(
                ObjectiveSpec,
                payload["objective"],
                enums={"time_sampling": TimeSampling},
            ),
            optimizer=_build_dataclass(OptimizerSpec, payload["optimizer"]),
            training=_build_dataclass(TrainingSpec, payload["training"]),
            checkpoint=_build_dataclass(
                CheckpointSpec,
                payload["checkpoint"],
                enums={"format": CheckpointFormat},
            ),
            topology=_build_dataclass(TopologySpec, payload["topology"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported training schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not _RUN_ID.fullmatch(self.run.run_id):
            raise ValueError("run.run_id contains unsupported characters")
        if not self.run.workdir:
            raise ValueError("run.workdir must not be empty")
        if self.run.seed < 0:
            raise ValueError("run.seed must be non-negative")

        model = self.model
        if (model.vocab_size, model.hidden_size, model.num_layers, model.expert_hidden_size) != (
            262_144,
            2_816,
            30,
            704,
        ):
            raise ValueError("model dimensions differ from the audited DiffusionGemma backbone")
        if model.num_experts <= 0 or not 0 < model.top_k_experts <= model.num_experts:
            raise ValueError("model.top_k_experts must be in [1, model.num_experts]")
        if model.dtype not in {"bfloat16", "float32"}:
            raise ValueError("model.dtype must be bfloat16 or float32")

        data = self.data
        if not _SHA256.fullmatch(data.manifest_sha256):
            raise ValueError("data.manifest_sha256 must be a lowercase SHA-256 digest")
        for name in ("global_batch_size", "prompt_length", "canvas_size", "num_canvases"):
            if getattr(data, name) <= 0:
                raise ValueError(f"data.{name} must be positive")
        if data.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
        for name in ("pad_token", "eos_token"):
            value = getattr(data, name)
            if not 0 <= value < model.vocab_size:
                raise ValueError(f"data.{name} is outside the vocabulary")

        objective = self.objective
        if not 1 <= objective.noise_draws <= 16:
            raise ValueError("objective.noise_draws must be in [1, 16]")
        if not 0.0 <= objective.self_condition_probability <= 1.0:
            raise ValueError("objective.self_condition_probability must be in [0, 1]")
        if objective.decoder_loss_weight < 0 or objective.encoder_loss_weight < 0:
            raise ValueError("objective loss weights must be non-negative")
        if objective.decoder_loss_weight + objective.encoder_loss_weight <= 0:
            raise ValueError("at least one objective loss must be enabled")
        if not 0.0 < objective.safety_epsilon < 0.5:
            raise ValueError("objective.safety_epsilon must be in (0, 0.5)")
        if objective.logit_normal_scale <= 0:
            raise ValueError("objective.logit_normal_scale must be positive")

        optimizer = self.optimizer
        if optimizer.peak_learning_rate <= 0 or optimizer.end_learning_rate < 0:
            raise ValueError("optimizer learning rates are invalid")
        if optimizer.end_learning_rate > optimizer.peak_learning_rate:
            raise ValueError("optimizer.end_learning_rate exceeds the peak")
        if optimizer.warmup_steps < 0:
            raise ValueError("optimizer.warmup_steps must be non-negative")
        if optimizer.weight_decay < 0 or optimizer.gradient_clip_norm <= 0:
            raise ValueError("optimizer decay/clip values are invalid")
        if not 0.0 <= optimizer.adam_beta1 < 1.0 or not 0.0 <= optimizer.adam_beta2 < 1.0:
            raise ValueError("optimizer Adam betas must be in [0, 1)")
        if optimizer.adam_epsilon <= 0 or optimizer.lora_rank <= 0:
            raise ValueError("optimizer epsilon and LoRA rank must be positive")

        training = self.training
        if training.steps <= 0:
            raise ValueError("training.steps must be positive")
        if self.run.phase is Phase.SMOKE and not 100 <= training.steps <= 500:
            raise ValueError("the readiness smoke must run 100-500 updates")
        if not 0 <= optimizer.warmup_steps < training.steps:
            raise ValueError("optimizer.warmup_steps must be smaller than training.steps")
        if not 1 <= training.checkpoint_every_steps <= training.steps:
            raise ValueError("training.checkpoint_every_steps is outside the run")
        if training.max_checkpoints_to_keep <= 0:
            raise ValueError("training.max_checkpoints_to_keep must be positive")
        for name in ("log_metrics_every_steps", "log_summaries_every_steps"):
            if getattr(training, name) <= 0:
                raise ValueError(f"training.{name} must be positive")

        if self.checkpoint.format is CheckpointFormat.EXACT_TREE:
            if self.checkpoint.init_step != -1:
                raise ValueError(
                    "checkpoint.init_step must be -1 for an exact-tree seed"
                )
        elif self.checkpoint.format is CheckpointFormat.KAULDRON_PARAMS:
            if self.checkpoint.init_step < 0:
                raise ValueError(
                    "checkpoint.init_step must pin an explicit Kauldron step"
                )
        if not self.checkpoint.init_path:
            raise ValueError("checkpoint.init_path must not be empty")
        if not self.checkpoint.model_param_path:
            raise ValueError("checkpoint.model_param_path must not be empty")

        topology = self.topology
        for name in ("expected_devices", "expected_processes", "expected_local_devices"):
            if getattr(topology, name) < 0:
                raise ValueError(f"topology.{name} must be non-negative")
        if topology.require_tpu:
            if topology.expected_devices <= 0 or topology.expected_processes <= 0:
                raise ValueError("TPU runs require expected device and process counts")
            if not self.run.workdir.startswith("gs://"):
                raise ValueError("TPU run.workdir must be a gs:// path")
        if topology.expected_processes and data.global_batch_size % topology.expected_processes:
            raise ValueError("data.global_batch_size must divide evenly across processes")
        if topology.expected_devices and data.global_batch_size % topology.expected_devices:
            raise ValueError("data.global_batch_size must divide evenly across devices")

    def canonical_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _expect_keys(
    context: str,
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} must be a table")
    missing = required - set(payload)
    unknown = set(payload) - required - optional
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")


def _build_dataclass(
    cls: type[Any],
    payload: Mapping[str, Any],
    *,
    enums: Mapping[str, type[enum.Enum]] = {},
) -> Any:
    fields = {field.name: field for field in dataclasses.fields(cls)}
    required = {
        name
        for name, field in fields.items()
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING  # type: ignore[comparison-overlap]
    }
    _expect_keys(cls.__name__, payload, required=required, optional=set(fields) - required)
    values = dict(payload)
    for name, enum_cls in enums.items():
        if name in values:
            try:
                values[name] = enum_cls(values[name])
            except (TypeError, ValueError) as error:
                choices = ", ".join(member.value for member in enum_cls)
                raise ValueError(f"{cls.__name__}.{name} must be one of {choices}") from error
    return cls(**values)


def _jsonable(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
