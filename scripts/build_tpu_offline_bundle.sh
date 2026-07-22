#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

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

bootstrap_python="${SUNFISH_BUILD_BOOTSTRAP_PYTHON:-python3}"
"${bootstrap_python}" -c \
  'import platform,sys,tarfile,urllib.request; assert sys.version_info >= (3,10); assert platform.python_implementation() == "CPython"; assert platform.system() == "Linux"; assert platform.machine().lower() in {"x86_64", "amd64"}; assert platform.libc_ver()[0].lower() == "glibc"; assert callable(tarfile.open); assert callable(urllib.request.urlopen)' || {
  echo "offline TPU bundles require stock CPython >=3.10 on glibc Linux x86_64" >&2
  exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT
bundle_root="${temporary}/sunfish-tpu-offline"
wheelhouse="${bundle_root}/wheelhouse"
runtime_archive_name='cpython-3.12.13+20260623-x86_64-unknown-linux-gnu-install_only.tar.gz'
runtime_archive="${bundle_root}/python-runtime/${runtime_archive_name}"
runtime_url='https://github.com/astral-sh/python-build-standalone/releases/download/20260623/cpython-3.12.13%2B20260623-x86_64-unknown-linux-gnu-install_only.tar.gz'
mkdir -p "${wheelhouse}" "$(dirname "${runtime_archive}")"

# This is the only public-network fetch beyond the pinned Python package
# resolver, and it runs only on the explicitly confirmed connected builder.
# The URL is never copied into bundle metadata or a TPU worker command.
"${bootstrap_python}" -c \
  'import pathlib,shutil,sys,urllib.request; url=sys.argv[1]; output=pathlib.Path(sys.argv[2]); request=urllib.request.Request(url,headers={"User-Agent":"sunfish-offline-builder/1"}); response=urllib.request.urlopen(request,timeout=60); destination=output.open("xb"); shutil.copyfileobj(response,destination,1024*1024); destination.close(); response.close()' \
  "${runtime_url}" "${runtime_archive}"
PYTHONPATH=src "${bootstrap_python}" -m sunfish_tpu.standalone_runtime write-metadata \
  --archive "${runtime_archive}" \
  --output "${bundle_root}/python-runtime.json"
builder_runtime_root="${temporary}/builder-runtime"
PYTHONPATH=src "${bootstrap_python}" -m sunfish_tpu.standalone_runtime install \
  --bundle-root "${bundle_root}" \
  --destination "${builder_runtime_root}"
PYTHON_BIN="${builder_runtime_root}/python/bin/python3"
"${PYTHON_BIN}" -c \
  'import platform,sys; assert platform.python_implementation()=="CPython"; assert platform.python_version()=="3.12.13"; assert platform.system()=="Linux"; assert platform.machine().lower() in {"x86_64","amd64"}; assert platform.libc_ver()[0].lower()=="glibc"' || {
  echo "downloaded standalone runtime is not exact CPython 3.12.13 for glibc Linux x86_64" >&2
  exit 2
}

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
  --no-binary=promise,sqlalchemy \
  --requirement requirements-tpu-base.lock
# Two transitive dependencies publish no usable wheel: promise (pure Python,
# via kauldron -> tensorflow-datasets) and sqlalchemy 1.2.19 (hard-pinned by
# every xmanager release; native extensions). Build those wheels here and drop
# the sdists — nothing unbuilt enters the bundle. The sqlalchemy wheel is
# ABI-bound to this builder's glibc, so the builder must run on a host whose
# image matches the TPU workers (enforced operationally, per infra/tpu/README).
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${wheelhouse}" \
  --no-deps \
  "${wheelhouse}"/promise-*.tar.gz "${wheelhouse}"/[Ss][Qq][Ll][Aa]lchemy-*.tar.gz
rm "${wheelhouse}"/promise-*.tar.gz "${wheelhouse}"/[Ss][Qq][Ll][Aa]lchemy-*.tar.gz
# A locally built native wheel is tagged bare linux_x86_64, which the audit
# rightly rejects. The builder host's image matches the TPU workers (documented
# above), so its glibc version is the honest manylinux compatibility bound.
builder_glibc_tag="manylinux_$(ldd --version | awk 'NR==1{v=$NF; gsub(/\./, "_", v); print v}')_x86_64"
"${PYTHON_BIN}" -m pip install --quiet wheel
for built_native in "${wheelhouse}"/*-linux_x86_64.whl; do
  [[ -e "${built_native}" ]] || continue
  "${PYTHON_BIN}" -m wheel tags --platform-tag "${builder_glibc_tag}" --remove "${built_native}"
done
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${wheelhouse}" \
  --no-deps \
  --requirement requirements-gemma-source.lock
# PyPI etils 1.14.0 calls a private jax._src.prng attribute jax 0.10.2 no
# longer exposes; this exact upstream commit fixes it without bumping the
# version string, so it must be wheeled from source like Gemma rather than
# pinned by version (see requirements-etils-source.lock). A plain
# `pip install` of this commit into an environment that already has PyPI
# 1.14.0 installed no-ops because pip sees the version requirement as already
# satisfied; --force-reinstall is required in that ad hoc case. It is not
# needed here: requirements-tpu-base.lock no longer downloads a PyPI etils
# wheel at all, so this is the only etils wheel that ever reaches the
# wheelhouse, and the later --no-index installs (bootstrap_tpu.sh,
# verify_tpu_bundled_runtime.sh --install) always build a brand-new venv with
# no prior etils installed to conflict with.
"${PYTHON_BIN}" -m pip wheel \
  --wheel-dir "${wheelhouse}" \
  --no-deps \
  --requirement requirements-etils-source.lock
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
# Isolated mode removes the checkout and PYTHONPATH from import resolution.
# Prove the regular wheel contains the exact config path used by sunfish-train.
"${second_venv}/bin/python" -I -c \
  'from sunfish_tpu.training.train import _packaged_kauldron_config_path; print(_packaged_kauldron_config_path())'
SUNFISH_OFFLINE_BUNDLE_MANIFEST="${bundle_root}/offline-bundle.json" \
  "${second_venv}/bin/sunfish-runtime-api-audit" \
  --output "${temporary}/runtime-api-audit.json"

PYTHONPATH=src "${PYTHON_BIN}" -m sunfish_tpu.offline_bundle pack \
  --bundle-root "${bundle_root}" \
  --output "${output}"
