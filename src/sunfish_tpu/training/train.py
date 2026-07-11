"""Validated distributed entrypoint for the Sunfish training harness."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from sunfish_tpu import kauldron_launch
from sunfish_tpu.training.spec import HarnessConfig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="strict Sunfish run TOML")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the stdlib-only contract without touching JAX",
    )
    parser.add_argument(
        "--allow-non-tpu",
        action="store_true",
        help="override topology checks for a single-process CPU development run",
    )
    parser.add_argument("kauldron_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = HarnessConfig.load(config_path)
    summary = {
        "ready_for_runtime_validation": True,
        "config": str(config_path),
        "config_sha256": config.digest,
        "run_id": config.run.run_id,
        "phase": config.run.phase.value,
        "dataset_manifest_sha256": config.data.manifest_sha256,
        "workdir": config.run.workdir,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return

    forwarded = list(args.kauldron_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if any(argument == "--cfg" or argument.startswith("--cfg=") or argument.startswith("--cfg.") for argument in forwarded):
        raise SystemExit(
            "Kauldron config overrides are disabled because they bypass the run-identity digest; edit the TOML"
        )

    os.environ["SUNFISH_TRAIN_CONFIG"] = str(config_path)
    config_module = (
        Path(__file__).resolve().parents[3] / "configs/training/sunfish.py"
    )
    allow_non_tpu = args.allow_non_tpu or not config.topology.require_tpu
    os.environ["SUNFISH_ALLOW_NON_TPU"] = "1" if allow_non_tpu else "0"
    expected_devices = 0 if allow_non_tpu else config.topology.expected_devices
    expected_processes = 0 if allow_non_tpu else config.topology.expected_processes
    expected_local_devices = (
        0 if allow_non_tpu else config.topology.expected_local_devices
    )
    launch_args = [
        "--expected-devices",
        str(expected_devices),
        "--expected-processes",
        str(expected_processes),
        "--expected-local-devices",
        str(expected_local_devices),
    ]
    if allow_non_tpu:
        launch_args.append("--allow-non-tpu")
    launch_args.extend(["--", f"--cfg={config_module}", *forwarded])
    print(json.dumps(summary, indent=2), flush=True)
    kauldron_launch.main(launch_args)


if __name__ == "__main__":
    main()
