#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: verify_tpu_bundled_runtime.sh --bundle-root ABSOLUTE_PATH [--install] [--expected-commit SHA] [--expected-tree SHA256]" >&2
}

bundle_root="${SUNFISH_OFFLINE_BUNDLE_ROOT:-}"
install=""
expected_commit=""
expected_tree=""
while (($#)); do
  case "$1" in
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --install) install=1; shift ;;
    --expected-commit) expected_commit="${2:-}"; shift 2 ;;
    --expected-tree) expected_tree="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "${bundle_root}" == /* && "${bundle_root}" != "/" && -d "${bundle_root}/source/src" ]] || {
  echo "bundled runtime verification requires an extracted absolute bundle root" >&2
  exit 2
}
[[ -z "${expected_commit}" || "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid expected source commit" >&2
  exit 2
}
[[ -z "${expected_tree}" || "${expected_tree}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid expected source-tree SHA-256" >&2
  exit 2
}

stock_python="${SUNFISH_STOCK_PYTHON_BIN:-python3}"
runtime_python="${SUNFISH_REMOTE_PYTHON_BIN:-${bundle_root}/python/bin/python3}"
runtime_command=(verify-installed)
if [[ "${install}" == 1 ]]; then
  runtime_command=(install)
fi
PYTHONPATH="${bundle_root}/source/src" \
  "${stock_python}" -m sunfish_tpu.standalone_runtime "${runtime_command[@]}" \
    --bundle-root "${bundle_root}" \
    --require-bundle-manifest

verify=(
  "${runtime_python}" -m sunfish_tpu.offline_bundle verify
  --bundle-root "${bundle_root}"
  --require-worker-runtime
)
if [[ -n "${expected_commit}" ]]; then
  verify+=(--expected-commit "${expected_commit}")
fi
if [[ -n "${expected_tree}" ]]; then
  verify+=(--expected-tree "${expected_tree}")
fi
PYTHONPATH="${bundle_root}/source/src" "${verify[@]}"
