"""Initialize distributed JAX, validate topology, then import Kauldron.

Never launch a pod job with ``python -m kauldron.main`` directly.  Kauldron
inspects JAX devices while its main module is imported, so distributed
initialization must happen first.  Arguments after ``--`` are forwarded
unchanged to ``kauldron.main``.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from collections.abc import Sequence

from sunfish_tpu.tpu_preflight import _jax_checks, report


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-non-tpu",
        action="store_true",
        help="allow the single-process CPU/GPU development fallback",
    )
    parser.add_argument(
        "--expected-devices",
        type=int,
        default=_env_int("EXPECTED_TPU_DEVICES", 0),
    )
    parser.add_argument(
        "--expected-processes",
        type=int,
        default=_env_int("EXPECTED_TPU_PROCESSES", 0),
    )
    parser.add_argument(
        "--expected-local-devices",
        type=int,
        default=_env_int("EXPECTED_LOCAL_TPU_DEVICES", 0),
    )
    parser.add_argument("kauldron_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = list(args.kauldron_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    # This call imports JAX, initializes the process cluster, and only then
    # imports jax.numpy or asks the backend for devices.
    checks = _jax_checks(
        require_tpu=not args.allow_non_tpu,
        require_distributed=not args.allow_non_tpu,
        expected_devices=args.expected_devices,
        expected_processes=args.expected_processes,
        expected_local_devices=args.expected_local_devices,
    )
    payload = report(checks)
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["ready"]:
        raise SystemExit("distributed JAX topology failed; Kauldron was not imported")

    # runpy preserves the behavior of `python -m kauldron.main`, but the
    # import now occurs strictly after distributed initialization.
    sys.argv = ["kauldron.main", *forwarded]
    runpy.run_module("kauldron.main", run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
