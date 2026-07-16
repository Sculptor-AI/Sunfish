#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: deploy_tpu_offline_bundle.sh --bundle ABSOLUTE_PATH.tar --remote-dir ABSOLUTE_PATH" >&2
}

bundle=""
remote_dir=""
while (($#)); do
  case "$1" in
    --bundle) bundle="${2:-}"; shift 2 ;;
    --remote-dir) remote_dir="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "${bundle}" == /* && -s "${bundle}" ]] || {
  echo "offline bundle must be an existing nonempty absolute path" >&2
  exit 2
}
[[ "${remote_dir}" == /* && "${remote_dir}" != "/" && "${remote_dir}" != */ && \
   "${remote_dir}" != *"/../"* && "${remote_dir}" != *"/.." && \
   "${remote_dir}" != *$'\n'* ]] || {
  echo "remote bundle directory must be a normalized non-root absolute path" >&2
  exit 2
}
sidecar="${bundle}.sha256"
[[ -f "${sidecar}" ]] || { echo "missing bundle SHA-256 sidecar" >&2; exit 2; }
read -r expected_sha sidecar_name extra < "${sidecar}"
[[ "${expected_sha}" =~ ^[0-9a-f]{64}$ && "${sidecar_name}" == "$(basename "${bundle}")" && -z "${extra:-}" ]] || {
  echo "invalid bundle SHA-256 sidecar" >&2
  exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
iap="${root}/scripts/tpu_iap.sh"
controller_python="${SUNFISH_CONTROLLER_PYTHON:-python3}"
stock_python="${SUNFISH_STOCK_PYTHON_BIN:-python3}"
git_commit="$(git -C "${root}" rev-parse HEAD)"
source_tree_sha256="$("${controller_python}" "${root}/scripts/source_tree_digest.py" --root "${root}")"
[[ "${git_commit}" =~ ^[0-9a-f]{40}$ && "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to resolve controller source identity" >&2
  exit 2
}

quote_arg() { printf '%q' "$1"; }
remote_temp="${remote_dir}.upload-${expected_sha:0:16}"
remote_archive="${remote_temp}/bundle.tar"
upload_helper_path="${root}/src/sunfish_tpu/upload_transaction.py"
upload_helper_b64="$("${controller_python}" -c \
  'import base64,pathlib,sys; print(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode("ascii"))' \
  "${upload_helper_path}")"
upload_helper_bootstrap='import base64,sys; source=base64.b64decode(sys.argv.pop(1),validate=True); exec(compile(source,"sunfish-upload-transaction.py","exec"))'
upload_helper="$(quote_arg "${stock_python}") -c $(quote_arg "${upload_helper_bootstrap}") $(quote_arg "${upload_helper_b64}")"
runtime_helper_path="${root}/src/sunfish_tpu/standalone_runtime.py"
runtime_helper_b64="$("${controller_python}" -c \
  'import base64,pathlib,sys; print(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode("ascii"))' \
  "${runtime_helper_path}")"
runtime_helper_bootstrap='import base64,sys; source=base64.b64decode(sys.argv.pop(1),validate=True); exec(compile(source,"sunfish-standalone-runtime.py","exec"))'
runtime_helper="$(quote_arg "${stock_python}") -c $(quote_arg "${runtime_helper_bootstrap}") $(quote_arg "${runtime_helper_b64}")"

prepare="${upload_helper} prepare"
prepare+=" --final $(quote_arg "${remote_dir}")"
prepare+=" --identity $(quote_arg "${expected_sha}")"
"${iap}" ssh-all --command "${prepare}"
"${iap}" scp-all "${bundle}" "${remote_archive}"

extracted="${remote_temp}/sunfish-tpu-offline"
finalize="${upload_helper} verify-file"
finalize+=" --final $(quote_arg "${remote_dir}")"
finalize+=" --identity $(quote_arg "${expected_sha}")"
finalize+=" --name bundle.tar --sha256 $(quote_arg "${expected_sha}")"
finalize+=" && ${runtime_helper} extract-bundle"
finalize+=" --archive $(quote_arg "${remote_archive}")"
finalize+=" --destination $(quote_arg "${remote_temp}")"
finalize+=" && cd $(quote_arg "${extracted}/source")"
finalize+=" && SUNFISH_STOCK_PYTHON_BIN=$(quote_arg "${stock_python}") scripts/verify_tpu_bundled_runtime.sh"
finalize+=" --bundle-root $(quote_arg "${extracted}") --install"
finalize+=" --expected-commit $(quote_arg "${git_commit}")"
finalize+=" --expected-tree $(quote_arg "${source_tree_sha256}")"
finalize+=" && if test -e $(quote_arg "${remote_dir}") || test -L $(quote_arg "${remote_dir}"); then"
finalize+=" test -d $(quote_arg "${remote_dir}") && test ! -L $(quote_arg "${remote_dir}")"
finalize+=" && cd $(quote_arg "${remote_dir}/source")"
finalize+=" && SUNFISH_STOCK_PYTHON_BIN=$(quote_arg "${stock_python}") scripts/verify_tpu_bundled_runtime.sh"
finalize+=" --bundle-root $(quote_arg "${remote_dir}")"
finalize+=" --expected-commit $(quote_arg "${git_commit}")"
finalize+=" --expected-tree $(quote_arg "${source_tree_sha256}");"
finalize+=" else mv $(quote_arg "${extracted}") $(quote_arg "${remote_dir}"); fi"
finalize+=" && ${upload_helper} cleanup"
finalize+=" --final $(quote_arg "${remote_dir}")"
finalize+=" --identity $(quote_arg "${expected_sha}")"
"${iap}" ssh-all --command "${finalize}"

echo "offline TPU bundle published on every worker: ${remote_dir}"
echo "set REMOTE_REPO_DIR=${remote_dir}/source"
echo "set SUNFISH_OFFLINE_BUNDLE_ROOT=${remote_dir}"
echo "bundled worker Python: ${remote_dir}/python/bin/python3"
