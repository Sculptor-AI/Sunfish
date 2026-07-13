#!/usr/bin/env bash
set -euo pipefail

# This environment loads a 25B checkpoint in float32 and is for a high-memory
# CPU/compute host only. Do not run it on Chase's laptop.
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
