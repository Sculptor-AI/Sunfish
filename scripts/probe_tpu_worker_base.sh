#!/usr/bin/env bash
set -euo pipefail

# Backend-free, network-free probe run before building/deploying the archive.
# It proves the immutable base image already has the interpreter primitives
# required by the air-gapped bootstrap; Sunfish never apt-installs on workers.

remote_python="${SUNFISH_REMOTE_PYTHON_BIN:-python3.12}"
min_free_bytes="${SUNFISH_MIN_FREE_BYTES:-21474836480}"
[[ "${min_free_bytes}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SUNFISH_MIN_FREE_BYTES must be a positive integer" >&2
  exit 2
}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/preflight_tpu_controller.sh"
log_dir="${SUNFISH_CONTROLLER_LOG_DIR:-tpu-launch-logs}/base-image-probe"
mkdir -p "${log_dir}"

quote_arg() { printf '%q' "$1"; }
program='import ensurepip,json,platform,shutil,sys,tarfile,venv; assert sys.version_info[:2]==(3,12); assert platform.system()=="Linux"; assert platform.machine().lower() in {"x86_64","amd64"}; libc=platform.libc_ver(); assert libc[0].lower()=="glibc"; bundled_pip=ensurepip.version(); assert bundled_pip; assert callable(getattr(tarfile,"data_filter",None)); minimum=int(sys.argv[1]); d=shutil.disk_usage(str(__import__("pathlib").Path.home())); assert d.free>=minimum,(d.free,minimum); print(json.dumps({"python":platform.python_version(),"machine":platform.machine(),"libc":libc,"ensurepip_version":bundled_pip,"free_bytes":d.free,"minimum_free_bytes":minimum},sort_keys=True))'
remote="$(quote_arg "${remote_python}") -c $(quote_arg "${program}") $(quote_arg "${min_free_bytes}")"
"${script_dir}/tpu_iap.sh" ssh-all --command "${remote}" \
  2>&1 | tee "${log_dir}/all-workers.log"
