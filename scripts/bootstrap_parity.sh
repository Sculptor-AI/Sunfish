#!/usr/bin/env bash
set -euo pipefail

# CONNECTED high-memory CPU host only, never a TPU worker. This environment
# loads a 25B checkpoint in float32. Do not run it on Chase's laptop.

[[ -z "${SUNFISH_TPU_WORKER:-}" ]] || {
  echo "Stage-0 parity setup is forbidden on a TPU worker" >&2
  exit 2
}
[[ "${1:-}" == "--connected-compute-host" && $# == 1 ]] || {
  echo "usage: bootstrap_parity.sh --connected-compute-host" >&2
  exit 2
}
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-parity}"
ENVIRONMENT_RECORD="${ENVIRONMENT_RECORD:-parity-environment.txt}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --requirement requirements-parity.lock
"${VENV_DIR}/bin/python" -m pip install --no-deps --editable .
"${VENV_DIR}/bin/python" -m pip freeze > "${ENVIRONMENT_RECORD}"
"${VENV_DIR}/bin/sunfish-parity" --help
cat "${ENVIRONMENT_RECORD}"
