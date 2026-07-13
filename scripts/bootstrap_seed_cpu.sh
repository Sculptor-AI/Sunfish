#!/usr/bin/env bash
set -euo pipefail

# High-memory Linux CPU VM only. The seed materializer has its own RAM guard
# and will refuse Chase's laptop even if this environment is created there.
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-seed}"
ENVIRONMENT_RECORD="${ENVIRONMENT_RECORD:-seed-environment.txt}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --requirement requirements-tpu-base.lock
"${VENV_DIR}/bin/python" -m pip install --no-deps --requirement requirements-gemma-source.lock
"${VENV_DIR}/bin/python" -m pip install --no-deps --editable .
"${VENV_DIR}/bin/python" -m pip freeze > "${ENVIRONMENT_RECORD}"
JAX_PLATFORMS=cpu "${VENV_DIR}/bin/sunfish-orbax-seed" --help
cat "${ENVIRONMENT_RECORD}"
