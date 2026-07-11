#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-tpu}"
: "${EXPECTED_TPU_DEVICES:?set EXPECTED_TPU_DEVICES to the granted global JAX device count}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -e '.[tpu]'

preflight=(
  "${VENV_DIR}/bin/sunfish-tpu-preflight"
  --require-tpu
  --expected-devices "${EXPECTED_TPU_DEVICES}"
)

if [[ -n "${SUNFISH_GCS_WORKDIR:-}" ]]; then
  preflight+=(
    --gcs-workdir "${SUNFISH_GCS_WORKDIR}"
    --require-gcs
    --probe-gcs-read
  )
fi

"${preflight[@]}"
"${VENV_DIR}/bin/python" -m pip freeze
