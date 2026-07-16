#!/usr/bin/env bash
set -euo pipefail

# CONTROLLER ONLY. TPU workers use bootstrap_tpu.sh with an offline wheelhouse.
[[ -z "${SUNFISH_TPU_WORKER:-}" ]] || {
  echo "controller bootstrap is forbidden on a TPU worker" >&2
  exit 2
}
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-tpu-controller}"
SUNFISH_CONTROLLER_PYTHON="${PYTHON_BIN}" \
  scripts/preflight_tpu_controller.sh --local-only
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --requirement requirements-controller.lock
"${VENV_DIR}/bin/python" -m pip install --no-deps --editable .
"${VENV_DIR}/bin/python" - <<'PY'
from etils import epath
from sunfish_tpu import (
    calibration_data_inventory,
    deployment_config,
    gcs_inventory,
    offline_bundle,
    parity_evidence,
    preemption_gate,
    readiness_ledger,
    smoke_evidence,
)

assert str(epath.Path("gs://example/sunfish")).startswith("gs://")
assert callable(preemption_gate.main)
assert callable(readiness_ledger.main)
assert callable(smoke_evidence.main)
assert callable(gcs_inventory.main)
assert callable(calibration_data_inventory.main)
assert callable(deployment_config.main)
assert callable(offline_bundle.verify_bundle)
assert callable(parity_evidence.validate_stage0_parity_report)
assert callable(readiness_ledger.validate_readiness_unlock)
print("Sunfish TPU controller environment ready")
PY
