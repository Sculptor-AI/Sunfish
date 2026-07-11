"""Write and exactly restore a small Orbax checkpoint on local disk or GCS.

Run this once on the actual TPU topology and GCS prefix before a long job. The
output path must not already exist and is intentionally retained as evidence
that write, finalization, and restore all succeeded.

This module is deliberately outside the dependency-free ``sunfish`` core.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _tree_equal(left: Any, right: Any) -> bool:
    import jax
    import numpy as np

    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    return left_structure == right_structure and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def run_smoke(workdir: str, run_id: str) -> dict[str, object]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, numbers, dot, underscore, or dash")

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from etils import epath

    state = {
        "step": jnp.asarray(137, dtype=jnp.int32),
        "params": {"probe": jnp.arange(64, dtype=jnp.bfloat16).reshape(8, 8)},
        "optimizer": {"count": jnp.asarray(137), "moment": jnp.linspace(0, 1, 8)},
        "data_cursor": jnp.asarray([12, 3456], dtype=jnp.int64),
        "rng": jnp.asarray([0x12345678, 0x9ABCDEF0], dtype=jnp.uint32),
    }
    abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
    destination = epath.Path(workdir) / "sunfish-checkpoint-smoke" / run_id

    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(destination, state)
        checkpointer.wait_until_finished()
        restored = checkpointer.restore(destination, abstract_state)
    exact = _tree_equal(state, restored)
    if not exact:
        raise RuntimeError("Orbax restored state does not exactly match saved state")
    return {
        "ready": True,
        "destination": str(destination),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "exact_restore": exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, help="local path or gs://bucket/prefix")
    parser.add_argument("--run-id", required=True, help="unique identifier; output is never overwritten")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.workdir, args.run_id), indent=2))


if __name__ == "__main__":
    main()
