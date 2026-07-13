#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-tpu}"
ENVIRONMENT_RECORD="${ENVIRONMENT_RECORD:-tpu-environment.txt}"
API_AUDIT_RECORD="${API_AUDIT_RECORD:-tpu-runtime-api-audit.json}"
: "${EXPECTED_TPU_DEVICES:?set EXPECTED_TPU_DEVICES to the granted global JAX device count}"
: "${EXPECTED_TPU_PROCESSES:?set EXPECTED_TPU_PROCESSES to the TPU host/process count}"
: "${SUNFISH_OFFLINE_BUNDLE_ROOT:?set SUNFISH_OFFLINE_BUNDLE_ROOT to the IAP-deployed bundle root}"
: "${SUNFISH_GIT_COMMIT:?bootstrap must run through the all-worker launcher}"
: "${SUNFISH_SOURCE_TREE_SHA256:?bootstrap must run through the all-worker launcher}"
[[ "${SUNFISH_TPU_WORKER:-}" == 1 ]] || {
  echo "TPU bootstrap must run through tpu_host_entrypoint.sh" >&2
  exit 2
}
[[ "${SUNFISH_OFFLINE_BUNDLE_ROOT}" == /* ]] || {
  echo "SUNFISH_OFFLINE_BUNDLE_ROOT must be absolute" >&2
  exit 2
}
[[ -n "${VENV_DIR}" && "${VENV_DIR}" != "/" ]] || {
  echo "VENV_DIR must name a non-root path" >&2
  exit 2
}

PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle verify \
  --bundle-root "${SUNFISH_OFFLINE_BUNDLE_ROOT}" \
  --expected-commit "${SUNFISH_GIT_COMMIT}" \
  --expected-tree "${SUNFISH_SOURCE_TREE_SHA256}"
complete_marker="${VENV_DIR}/.sunfish-offline-complete"
building_marker="${VENV_DIR}/.sunfish-offline-building"
if [[ -e "${VENV_DIR}" && (! -d "${VENV_DIR}" || -L "${VENV_DIR}") ]]; then
  echo "existing TPU environment is not a regular directory: ${VENV_DIR}" >&2
  exit 2
fi
for marker in "${complete_marker}" "${building_marker}"; do
  if [[ -e "${marker}" && (! -f "${marker}" || -L "${marker}") ]]; then
    echo "refusing unexpected TPU environment marker object: ${marker}" >&2
    exit 2
  fi
done
if [[ -f "${complete_marker}" && -e "${building_marker}" ]]; then
  echo "TPU environment has conflicting completion/building markers" >&2
  exit 2
fi
if [[ -d "${VENV_DIR}" && ! -f "${complete_marker}" ]]; then
  [[ -f "${building_marker}" ]] || {
    echo "refusing to remove an unmarked existing environment: ${VENV_DIR}" >&2
    exit 2
  }
  read -r building_commit building_tree building_extra < "${building_marker}"
  if [[ "${building_commit}" != "${SUNFISH_GIT_COMMIT}" || \
        "${building_tree}" != "${SUNFISH_SOURCE_TREE_SHA256}" || \
        -n "${building_extra:-}" ]]; then
    echo "incomplete TPU environment belongs to a different release" >&2
    exit 2
  fi
  # The exact release-scoped building marker proves this is our partial venv.
  rm -rf -- "${VENV_DIR}"
fi
if [[ ! -e "${VENV_DIR}" ]]; then
  mkdir "${VENV_DIR}"
  printf '%s %s\n' \
    "${SUNFISH_GIT_COMMIT}" \
    "${SUNFISH_SOURCE_TREE_SHA256}" > "${building_marker}"
  creation_in_progress=1
  cleanup_incomplete_environment() {
    if [[ -n "${creation_in_progress:-}" && -d "${VENV_DIR}" && ! -L "${VENV_DIR}" ]]; then
      rm -rf -- "${VENV_DIR}"
    fi
  }
  trap cleanup_incomplete_environment EXIT
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  PIP_CONFIG_FILE=/dev/null \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_INDEX=1 \
  "${VENV_DIR}/bin/python" -m pip install \
    --no-index \
    --no-deps \
    --only-binary=:all: \
    --find-links "${SUNFISH_OFFLINE_BUNDLE_ROOT}/wheelhouse" \
    --requirement "${SUNFISH_OFFLINE_BUNDLE_ROOT}/offline-requirements.lock"
  PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle verify-installed \
    --bundle-root "${SUNFISH_OFFLINE_BUNDLE_ROOT}" \
    --python "${VENV_DIR}/bin/python"
  mv "${building_marker}" "${complete_marker}"
  creation_in_progress=""
  trap - EXIT
fi
read -r marker_commit marker_tree marker_extra < "${complete_marker}"
if [[ "${marker_commit}" != "${SUNFISH_GIT_COMMIT}" || \
      "${marker_tree}" != "${SUNFISH_SOURCE_TREE_SHA256}" || \
      -n "${marker_extra:-}" ]]; then
  echo "TPU environment completion marker differs from this release" >&2
  exit 2
fi

# A controller retry can reuse a published environment only after proving the
# complete manifest/version contract again. Partial builds have no marker and
# are rebuilt in place so virtualenv entry-point shebangs remain valid.
PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle verify-installed \
  --bundle-root "${SUNFISH_OFFLINE_BUNDLE_ROOT}" \
  --python "${VENV_DIR}/bin/python"
PIP_NO_INDEX=1 "${VENV_DIR}/bin/python" -m pip check

# This inspection reads installed source text through importlib.metadata and
# deliberately does not import JAX, Gemma, Kauldron, or Orbax.  Fail before
# backend initialization if a load-bearing pinned/private API differs.
SUNFISH_OFFLINE_BUNDLE_MANIFEST="${SUNFISH_OFFLINE_BUNDLE_ROOT}/offline-bundle.json" \
  "${VENV_DIR}/bin/sunfish-runtime-api-audit" --output "${API_AUDIT_RECORD}"

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
