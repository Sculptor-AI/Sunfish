#!/usr/bin/env bash
set -euo pipefail

# CONNECTED high-memory Linux CPU VM only, never a TPU worker. The seed
# materializer has its own RAM guard and refuses Chase's laptop.

[[ -z "${SUNFISH_TPU_WORKER:-}" ]] || {
  echo "seed materialization is forbidden on a TPU worker" >&2
  exit 2
}
[[ "${1:-}" == "--connected-compute-host" && $# == 1 ]] || {
  echo "usage: bootstrap_seed_cpu.sh --connected-compute-host" >&2
  exit 2
}
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
