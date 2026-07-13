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
[[ "${remote_dir}" == /* && "${remote_dir}" != *$'\n'* ]] || {
  echo "remote config directory must be an absolute one-line path" >&2
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
PYTHONPATH=src "${controller_python}" -c \
  'import pathlib,sys; from sunfish_tpu.deployment_config import validate_rendered_config_file; validate_rendered_config_file(pathlib.Path(sys.argv[1]), source_root=pathlib.Path(sys.argv[2]), require_bundle=True)' \
  "${local_dir}/sunfish-smoke.toml" "$(pwd)"

quote_arg() { printf '%q' "$1"; }
gcloud_bin="${SUNFISH_GCLOUD_BIN:-gcloud}"
bundle_digest="$(python3 -c \
  'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "${local_dir}/rendered-configs.json")"
[[ "${bundle_digest}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to hash rendered config bundle" >&2
  exit 2
}
remote_parent="${remote_dir%/*}"
[[ -n "${remote_parent}" ]] || remote_parent="/"
remote_temp="${remote_dir}.upload-${bundle_digest:0:16}"
prepare="test ! -e $(quote_arg "${remote_dir}")"
prepare+=" && test ! -e $(quote_arg "${remote_temp}")"
prepare+=" && mkdir -p $(quote_arg "${remote_parent}")"
prepare+=" && mkdir $(quote_arg "${remote_temp}")"
"${gcloud_bin}" compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT_ID}" --zone "${ZONE}" --worker=all \
  --command "${prepare}"

for filename in "${files[@]}"; do
  local_path="${local_dir}/${filename}"
  remote_path="${remote_temp}/${filename}"
  "${gcloud_bin}" compute tpus tpu-vm scp "${local_path}" \
    "${TPU_NAME}:${remote_path}" \
    --project "${PROJECT_ID}" --zone "${ZONE}" --worker=all
  digest="$(python3 -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "${local_path}")"
  verify="python3 -c $(quote_arg 'import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); raise SystemExit(0 if actual == sys.argv[2] else 1)')"
  verify+=" $(quote_arg "${remote_path}") $(quote_arg "${digest}")"
  "${gcloud_bin}" compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project "${PROJECT_ID}" --zone "${ZONE}" --worker=all \
    --command "${verify}"
done

finalize="test ! -e $(quote_arg "${remote_dir}")"
finalize+=" && mv $(quote_arg "${remote_temp}") $(quote_arg "${remote_dir}")"
"${gcloud_bin}" compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT_ID}" --zone "${ZONE}" --worker=all \
  --command "${finalize}"
