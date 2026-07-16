#!/usr/bin/env bash
set -euo pipefail

# Backend-free, network-free probe run before building/deploying the archive.
# It proves only that the immutable base image can run the stock-Python 3.10+
# predeploy helpers. Exact CPython 3.12.13 is supplied by, and checked after,
# the offline bundle; Sunfish never apt-installs on workers.

stock_python="${SUNFISH_STOCK_PYTHON_BIN:-python3}"
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
program='import hashlib,json,os,platform,shutil,sys,tarfile,urllib.request; assert platform.python_implementation()=="CPython"; assert sys.version_info>=(3,10); assert platform.system()=="Linux"; assert platform.machine().lower() in {"x86_64","amd64"}; libc=platform.libc_ver(); assert libc[0].lower()=="glibc"; proxy_vars=("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"); proxy_names=sorted({n for n in proxy_vars if os.environ.get(n)}|{str(k) for k,v in urllib.request.getproxies().items() if str(k).lower() in {"http","https","all"} and v}); assert not proxy_names,("worker HTTP(S)/ALL proxy settings are forbidden",proxy_names); assert callable(tarfile.open) and callable(hashlib.sha256); minimum=int(sys.argv[1]); d=shutil.disk_usage(str(__import__("pathlib").Path.home())); assert d.free>=minimum,(d.free,minimum); print(json.dumps({"stock_python":platform.python_version(),"stock_python_compatible":True,"bundled_python_required":"3.12.13","machine":platform.machine(),"libc":libc,"proxy_environment_clear":True,"free_bytes":d.free,"minimum_free_bytes":minimum},sort_keys=True))'
remote="$(quote_arg "${stock_python}") -c $(quote_arg "${program}") $(quote_arg "${min_free_bytes}")"
"${script_dir}/tpu_iap.sh" ssh-all --command "${remote}" \
  2>&1 | tee "${log_dir}/all-workers.log"
