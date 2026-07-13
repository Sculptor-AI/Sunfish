"""Render one immutable Stage-0.5 config bundle from reviewed templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sunfish.source_tree import workspace_source_identity
from sunfish_tpu.parity_evidence import validate_stage0_parity_report
from sunfish_tpu.training.spec import HarnessConfig

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TEMPLATES = {
    "sunfish-smoke.toml": ("sunfish-stage05-smoke", "stage05-smoke"),
    "sunfish-resume-smoke.toml": (
        "sunfish-stage05-real-resume",
        "stage05-real-resume",
    ),
    "sunfish-preemption-smoke.toml": (
        "sunfish-stage05-preemption",
        "stage05-preemption",
    ),
}


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _replace_assignment(
    text: str, *, section: str, key: str, replacement: str
) -> str:
    lines = text.splitlines()
    current = ""
    matches = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            continue
        if current == section and re.match(rf"^{re.escape(key)}\s*=", stripped):
            lines[index] = f"{key} = {replacement}"
            matches += 1
    if matches != 1:
        raise ValueError(f"template has {matches} assignments for [{section}] {key}")
    return "\n".join(lines) + "\n"


def _render_one(
    template: str,
    *,
    run_id: str,
    workdir: str,
    storage_root: str,
    run_tag: str,
    dataset_manifest_sha256: str,
    seed_manifest_sha256: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> str:
    assignments: tuple[tuple[str, str, str], ...] = (
        ("run", "run_id", _toml_string(run_id)),
        ("run", "workdir", _toml_string(workdir)),
        (
            "data",
            "directory",
            _toml_string(f"{storage_root}/data/tiny-overfit-{run_tag}"),
        ),
        ("data", "manifest_sha256", _toml_string(dataset_manifest_sha256)),
        (
            "checkpoint",
            "init_path",
            _toml_string(
                f"{storage_root}/checkpoints/sunfish-stage05-first32-exact-tree"
            ),
        ),
        (
            "checkpoint",
            "init_manifest_path",
            _toml_string(
                f"{storage_root}/checkpoints/"
                "sunfish-stage05-first32-exact-tree.json"
            ),
        ),
        (
            "checkpoint",
            "init_manifest_sha256",
            _toml_string(seed_manifest_sha256),
        ),
        ("topology", "expected_devices", str(expected_devices)),
        ("topology", "expected_processes", str(expected_processes)),
        ("topology", "expected_local_devices", str(expected_local_devices)),
    )
    rendered = template
    for section, key, value in assignments:
        rendered = _replace_assignment(
            rendered, section=section, key=key, replacement=value
        )
    return rendered


def render_stage05_configs(
    *,
    template_directory: Path,
    output_directory: Path,
    storage_root: str,
    run_tag: str,
    dataset_manifest_sha256: str,
    seed_manifest_sha256: str,
    parity_report_path: Path,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
    source_root: Path,
) -> dict[str, Any]:
    """Render and validate all three isolated readiness run configs."""
    expected_template_directory = source_root.resolve() / "configs/training"
    if template_directory.resolve() != expected_template_directory:
        raise ValueError(
            "real readiness configs must use the source-bound configs/training templates"
        )
    storage_root = storage_root.rstrip("/")
    if (
        not storage_root.startswith("gs://")
        or len(storage_root.split("/", 3)) < 4
        or any(character.isspace() for character in storage_root)
        or "YOUR_BUCKET" in storage_root
    ):
        raise ValueError("storage_root must be a concrete gs://bucket/prefix")
    if not _TAG.fullmatch(run_tag):
        raise ValueError("run_tag contains unsupported characters")
    for name, value in (
        ("dataset_manifest_sha256", dataset_manifest_sha256),
        ("seed_manifest_sha256", seed_manifest_sha256),
    ):
        if not _SHA256.fullmatch(value) or value == "0" * 64:
            raise ValueError(f"{name} must be a nonzero lowercase SHA-256")
    if min(expected_devices, expected_processes, expected_local_devices) <= 0:
        raise ValueError("topology counts must be positive")
    if expected_processes * expected_local_devices != expected_devices:
        raise ValueError("processes * local devices must equal global devices")
    if output_directory.exists():
        raise FileExistsError(f"immutable config output exists: {output_directory}")
    source_identity = workspace_source_identity(source_root)
    parity_summary, parity_report_bytes = validate_stage0_parity_report(
        parity_report_path, expected_source=source_identity
    )

    rendered: dict[str, str] = {}
    configs: dict[str, HarnessConfig] = {}
    template_hashes: dict[str, str] = {}
    for filename, (run_prefix, workdir_suffix) in _TEMPLATES.items():
        template_path = template_directory / filename
        template_bytes = template_path.read_bytes()
        template_hashes[filename] = hashlib.sha256(template_bytes).hexdigest()
        text = _render_one(
            template_bytes.decode("utf-8"),
            run_id=f"{run_prefix}-{run_tag}",
            workdir=f"{storage_root}/runs/{workdir_suffix}-{run_tag}",
            storage_root=storage_root,
            run_tag=run_tag,
            dataset_manifest_sha256=dataset_manifest_sha256,
            seed_manifest_sha256=seed_manifest_sha256,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        )
        config = HarnessConfig.from_mapping(tomllib.loads(text))
        rendered[filename] = text
        configs[filename] = config

    baseline = configs["sunfish-smoke.toml"].canonical_dict()
    baseline["run"].pop("run_id")
    baseline["run"].pop("workdir")
    for filename, config in configs.items():
        comparable = config.canonical_dict()
        comparable["run"].pop("run_id")
        comparable["run"].pop("workdir")
        if comparable != baseline:
            raise ValueError(f"rendered readiness config drifted: {filename}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "stage-0.5-rendered-config-bundle",
        "run_tag": run_tag,
        "storage_root": storage_root,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "seed_manifest_sha256": seed_manifest_sha256,
        "stage0_parity": {
            "filename": "stage0-parity-report.json",
            **parity_summary,
        },
        "topology": {
            "global_devices": expected_devices,
            "processes": expected_processes,
            "local_devices": expected_local_devices,
        },
        "templates": template_hashes,
        "sunfish_source": source_identity,
        "configs": {
            filename: {
                "run_id": configs[filename].run.run_id,
                "workdir": configs[filename].run.workdir,
                "config_sha256": configs[filename].digest,
                "config_file_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            for filename, text in rendered.items()
        },
    }
    output_directory.mkdir(parents=True)
    for filename, text in rendered.items():
        (output_directory / filename).write_text(text, encoding="utf-8")
    (output_directory / "stage0-parity-report.json").write_bytes(
        parity_report_bytes
    )
    (output_directory / "rendered-configs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def validate_rendered_config_file(
    config_path: Path, *, source_root: Path, require_bundle: bool = False
) -> dict[str, Any] | None:
    """Validate one rendered config against its sibling bundle manifest."""
    manifest_path = config_path.parent / "rendered-configs.json"
    if not manifest_path.exists():
        if require_bundle:
            raise ValueError(
                "TPU launch requires a rendered config bundle with Stage-0 parity evidence"
            )
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("rendered config bundle manifest is invalid")
    if payload.get("purpose") != "stage-0.5-rendered-config-bundle":
        raise ValueError("rendered config bundle purpose is invalid")
    if payload.get("sunfish_source") != workspace_source_identity(source_root):
        raise ValueError("rendered config bundle belongs to a different source tree")
    configs_payload = payload.get("configs")
    if not isinstance(configs_payload, Mapping) or set(configs_payload) != set(
        _TEMPLATES
    ):
        raise ValueError("rendered config bundle config inventory differs")
    if config_path.name not in configs_payload:
        raise ValueError(f"config is not listed in rendered bundle: {config_path.name}")
    template_hashes = {
        filename: hashlib.sha256(
            (source_root / "configs/training" / filename).read_bytes()
        ).hexdigest()
        for filename in _TEMPLATES
    }
    if payload.get("templates") != template_hashes:
        raise ValueError("rendered config bundle template hashes differ")
    run_tag = payload.get("run_tag")
    storage_root = payload.get("storage_root")
    if not isinstance(run_tag, str) or _TAG.fullmatch(run_tag) is None:
        raise ValueError("rendered config bundle run tag is invalid")
    if not isinstance(storage_root, str) or not storage_root.startswith("gs://"):
        raise ValueError("rendered config bundle storage root is invalid")
    loaded: dict[str, HarnessConfig] = {}
    for filename, (run_prefix, workdir_suffix) in _TEMPLATES.items():
        rendered_path = config_path.parent / filename
        config = HarnessConfig.load(rendered_path)
        loaded[filename] = config
        actual = {
            "run_id": config.run.run_id,
            "workdir": config.run.workdir,
            "config_sha256": config.digest,
            "config_file_sha256": hashlib.sha256(
                rendered_path.read_bytes()
            ).hexdigest(),
        }
        if dict(configs_payload[filename]) != actual:
            raise ValueError(f"rendered config differs from bundle: {rendered_path}")
        if config.run.run_id != f"{run_prefix}-{run_tag}":
            raise ValueError(f"rendered config run ID differs: {filename}")
        if config.run.workdir != f"{storage_root}/runs/{workdir_suffix}-{run_tag}":
            raise ValueError(f"rendered config workdir differs: {filename}")
    dataset_hashes = {config.data.manifest_sha256 for config in loaded.values()}
    seed_hashes = {
        config.checkpoint.init_manifest_sha256 for config in loaded.values()
    }
    topologies = {
        (
            config.topology.expected_devices,
            config.topology.expected_processes,
            config.topology.expected_local_devices,
        )
        for config in loaded.values()
    }
    if dataset_hashes != {payload.get("dataset_manifest_sha256")}:
        raise ValueError("rendered config bundle dataset hashes differ")
    if seed_hashes != {payload.get("seed_manifest_sha256")}:
        raise ValueError("rendered config bundle seed hashes differ")
    expected_topology = payload.get("topology", {})
    if topologies != {
        (
            expected_topology.get("global_devices"),
            expected_topology.get("processes"),
            expected_topology.get("local_devices"),
        )
    }:
        raise ValueError("rendered config bundle topology differs")
    parity_pin = payload.get("stage0_parity")
    if not isinstance(parity_pin, Mapping):
        raise ValueError("rendered config bundle has no Stage-0 parity pin")
    parity_filename = parity_pin.get("filename")
    if parity_filename != "stage0-parity-report.json":
        raise ValueError("rendered config bundle parity filename is invalid")
    parity_path = config_path.parent / parity_filename
    parity_summary, _ = validate_stage0_parity_report(
        parity_path, expected_source=payload["sunfish_source"]
    )
    if dict(parity_pin) != {"filename": parity_filename, **parity_summary}:
        raise ValueError("rendered config bundle parity report differs")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", required=True, help="gs://bucket/sunfish")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--seed-manifest-sha256", required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template-dir", type=Path, default=Path("configs/training")
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        payload = render_stage05_configs(
            template_directory=args.template_dir,
            output_directory=args.output_dir,
            storage_root=args.storage_root,
            run_tag=args.run_tag,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
            seed_manifest_sha256=args.seed_manifest_sha256,
            parity_report_path=args.parity_report,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
            source_root=root,
        )
    except (FileExistsError, FileNotFoundError, UnicodeDecodeError, ValueError) as error:
        print(f"sunfish-render-tpu-configs: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
