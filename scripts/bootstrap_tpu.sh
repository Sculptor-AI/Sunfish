#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-tpu}"
ENVIRONMENT_RECORD="${ENVIRONMENT_RECORD:-tpu-environment.txt}"
: "${EXPECTED_TPU_DEVICES:?set EXPECTED_TPU_DEVICES to the granted global JAX device count}"
: "${EXPECTED_TPU_PROCESSES:?set EXPECTED_TPU_PROCESSES to the TPU host/process count}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --requirement requirements-tpu-base.lock
# Gemma 4.1.0 is not released yet. Its source metadata points at an unpinned
# Hackable Diffusion branch, so install the audited commit without dependencies
# after the exact base stack. Runtime validation checks direct_url.json too.
"${VENV_DIR}/bin/python" -m pip install --no-deps --requirement requirements-gemma-source.lock
"${VENV_DIR}/bin/python" -m pip install --no-deps --editable .

preflight=(
  "${VENV_DIR}/bin/sunfish-tpu-preflight"
  --require-tpu
  --require-distributed
  --expected-devices "${EXPECTED_TPU_DEVICES}"
  --expected-processes "${EXPECTED_TPU_PROCESSES}"
)

if [[ -n "${EXPECTED_LOCAL_TPU_DEVICES:-}" ]]; then
  preflight+=(--expected-local-devices "${EXPECTED_LOCAL_TPU_DEVICES}")
fi

if [[ -n "${SUNFISH_GCS_WORKDIR:-}" ]]; then
  preflight+=(
    --gcs-workdir "${SUNFISH_GCS_WORKDIR}"
    --require-gcs
    --probe-gcs-read
  )
fi

"${preflight[@]}"
"${VENV_DIR}/bin/python" -m pip freeze > "${ENVIRONMENT_RECORD}"
cat "${ENVIRONMENT_RECORD}"
