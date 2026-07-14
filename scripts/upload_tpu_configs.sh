#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: upload_tpu_configs.sh --local-dir PATH --remote-dir ABSOLUTE_PATH" >&2
}

local_dir=""
remote_dir=""
while (($#)); do
  case "$1" in
    --local-dir) local_dir="${2:-}"; shift 2 ;;
    --remote-dir) remote_dir="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -d "${local_dir}" ]] || { echo "local config directory not found" >&2; exit 2; }
[[ "${remote_dir}" == /* && "${remote_dir}" != "/" && "${remote_dir}" != */ && \
   "${remote_dir}" != *"/../"* && "${remote_dir}" != *"/.." && \
   "${remote_dir}" != *$'\n'* ]] || {
  echo "remote config directory must be a normalized non-root absolute path" >&2
  exit 2
}
: "${TPU_NAME:?set TPU_NAME}"
: "${PROJECT_ID:?set PROJECT_ID}"
: "${ZONE:?set ZONE}"

files=(
  sunfish-smoke.toml
  sunfish-resume-smoke.toml
  sunfish-preemption-smoke.toml
  stage0-parity-report.json
  rendered-configs.json
)
for filename in "${files[@]}"; do
  [[ -f "${local_dir}/${filename}" ]] || {
    echo "rendered config bundle is missing ${filename}" >&2
    exit 2
  }
done

controller_python="${SUNFISH_CONTROLLER_PYTHON:-python3}"
remote_python="${SUNFISH_REMOTE_PYTHON_BIN:-python3.12}"
PYTHONPATH=src "${controller_python}" -c \
  'import pathlib,sys; from sunfish_tpu.deployment_config import validate_rendered_config_file; validate_rendered_config_file(pathlib.Path(sys.argv[1]), source_root=pathlib.Path(sys.argv[2]), require_bundle=True)' \
  "${local_dir}/sunfish-smoke.toml" "$(pwd)"

quote_arg() { printf '%q' "$1"; }
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${script_dir}/.." && pwd)"
iap="${script_dir}/tpu_iap.sh"
bundle_digest="$("${controller_python}" -c \
  'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "${local_dir}/rendered-configs.json")"
[[ "${bundle_digest}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to hash rendered config bundle" >&2
  exit 2
}
remote_temp="${remote_dir}.upload-${bundle_digest:0:16}"
upload_helper_path="${root}/src/sunfish_tpu/upload_transaction.py"
upload_helper_b64="$("${controller_python}" -c \
  'import base64,pathlib,sys; print(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode("ascii"))' \
  "${upload_helper_path}")"
upload_helper_bootstrap='import base64,sys; source=base64.b64decode(sys.argv.pop(1),validate=True); exec(compile(source,"sunfish-upload-transaction.py","exec"))'
upload_helper="$(quote_arg "${remote_python}") -c $(quote_arg "${upload_helper_bootstrap}") $(quote_arg "${upload_helper_b64}")"
prepare="${upload_helper} prepare"
prepare+=" --final $(quote_arg "${remote_dir}")"
prepare+=" --identity $(quote_arg "${bundle_digest}")"
"${iap}" ssh-all --command "${prepare}"

file_specs=()
for filename in "${files[@]}"; do
  local_path="${local_dir}/${filename}"
  remote_path="${remote_temp}/${filename}"
  "${iap}" scp-all "${local_path}" "${remote_path}"
  digest="$("${controller_python}" -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "${local_path}")"
  file_specs+=("${filename}=${digest}")
  verify="${upload_helper} verify-file"
  verify+=" --final $(quote_arg "${remote_dir}")"
  verify+=" --identity $(quote_arg "${bundle_digest}")"
  verify+=" --name $(quote_arg "${filename}") --sha256 $(quote_arg "${digest}")"
  "${iap}" ssh-all --command "${verify}"
done

finalize="${upload_helper} publish-files"
finalize+=" --final $(quote_arg "${remote_dir}")"
finalize+=" --identity $(quote_arg "${bundle_digest}")"
for file_spec in "${file_specs[@]}"; do
  finalize+=" --file $(quote_arg "${file_spec}")"
done
"${iap}" ssh-all --command "${finalize}"
