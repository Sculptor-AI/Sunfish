#!/usr/bin/env bash
set -euo pipefail

# CONNECTED LINUX BUILD HOST ONLY (Colab/Kaggle/Cloud Build/CPU VM).
# TPU workers have no public egress and must never execute this script.

[[ -z "${SUNFISH_TPU_WORKER:-}" ]] || {
  echo "offline bundle construction is forbidden on a TPU worker" >&2
  exit 2
}

usage() {
  echo "usage: build_tpu_offline_bundle.sh --connected-build-host --output ABSOLUTE_PATH.tar" >&2
}

confirmed=""
output=""
while (($#)); do
  case "$1" in
    --connected-build-host) confirmed=1; shift ;;
    --output) output="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "${confirmed}" == 1 ]] || {
  echo "refusing network dependency resolution without --connected-build-host" >&2
  exit 2
}
[[ "${output}" == /* && "${output}" == *.tar ]] || {
  echo "--output must be an absolute .tar path" >&2
  exit 2
}
[[ ! -e "${output}" && ! -e "${output}.sha256" ]] || {
  echo "output or SHA-256 sidecar already exists" >&2
  exit 2
}

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"${PYTHON_BIN}" -c \
  'import platform,sys; assert sys.version_info[:2] == (3, 12); assert platform.system() == "Linux"; assert platform.machine().lower() in {"x86_64", "amd64"}; assert platform.libc_ver()[0].lower() == "glibc"' || {
  echo "offline TPU bundles must be built on glibc Linux x86_64 with CPython 3.12" >&2
  exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT
bundle_root="${temporary}/sunfish-tpu-offline"
wheelhouse="${bundle_root}/wheelhouse"
mkdir -p "${wheelhouse}"

PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle export-source \
  --repository "${root}" \
  --output "${bundle_root}/source"

# Network is allowed only here, on the explicitly confirmed packaging host.
# Workers receive the resulting wheel-only archive through IAP SCP.
# Third-party runtime artifacts must already be wheels for the target ABI. Do
# not silently compile an sdist against whatever happens to be on the builder.
"${PYTHON_BIN}" -m pip download \
  --dest "${wheelhouse}" \
  --only-binary=:all: \
  --requirement requirements-tpu-base.lock
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${wheelhouse}" \
  --no-deps \
  --requirement requirements-gemma-source.lock
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${wheelhouse}" \
  --no-deps \
  .

first_venv="${temporary}/first-validation"
"${PYTHON_BIN}" -m venv "${first_venv}"
wheel_files=("${wheelhouse}"/*.whl)
[[ -e "${wheel_files[0]}" ]] || {
  echo "wheelhouse is empty" >&2
  exit 2
}
PIP_CONFIG_FILE=/dev/null \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
PIP_NO_INDEX=1 \
"${first_venv}/bin/python" -m pip install \
  --no-index \
  --no-deps \
  --only-binary=:all: \
  "${wheel_files[@]}"
PIP_NO_INDEX=1 "${first_venv}/bin/python" -m pip check

PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle write-lock \
  --python "${first_venv}/bin/python" \
  --output "${bundle_root}/offline-requirements.lock"
PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle create-manifest \
  --bundle-root "${bundle_root}"

# Prove a fresh environment can be reconstructed without any index, URL,
# source checkout, build isolation, or network access.
second_venv="${temporary}/second-validation"
"${PYTHON_BIN}" -m venv "${second_venv}"
PIP_CONFIG_FILE=/dev/null \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
PIP_NO_INDEX=1 \
"${second_venv}/bin/python" -m pip install \
  --no-index \
  --no-deps \
  --only-binary=:all: \
  --find-links "${wheelhouse}" \
  --requirement "${bundle_root}/offline-requirements.lock"
PIP_NO_INDEX=1 "${second_venv}/bin/python" -m pip check
PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle verify-installed \
  --bundle-root "${bundle_root}" \
  --python "${second_venv}/bin/python"
SUNFISH_OFFLINE_BUNDLE_MANIFEST="${bundle_root}/offline-bundle.json" \
  "${second_venv}/bin/sunfish-runtime-api-audit" \
  --output "${temporary}/runtime-api-audit.json"

PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle pack \
  --bundle-root "${bundle_root}" \
  --output "${output}"
