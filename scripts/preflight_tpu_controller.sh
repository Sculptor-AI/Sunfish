#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: preflight_tpu_controller.sh [--local-only]" >&2
}

local_only=0
while (($#)); do
  case "$1" in
    --local-only) local_only=1; shift ;;
    *) usage; exit 2 ;;
  esac
done

[[ -z "${SUNFISH_TPU_WORKER:-}" ]] || {
  echo "controller preflight is forbidden on a TPU worker" >&2
  exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${script_dir}/.." && pwd)"
python_bin="${SUNFISH_CONTROLLER_PYTHON:-${PYTHON_BIN:-python3.12}}"
gcloud_bin="${SUNFISH_GCLOUD_BIN:-gcloud}"
ssh_add_bin="${SUNFISH_SSH_ADD_BIN:-ssh-add}"

command -v "${python_bin}" >/dev/null 2>&1 || {
  echo "CPython 3.12 is required on the controller: ${python_bin}" >&2
  exit 2
}
"${python_bin}" -c \
  'import platform,sys; raise SystemExit(0 if platform.python_implementation()=="CPython" and sys.version_info[:2] == (3, 12) else 1)' || {
  echo "controller interpreter must be CPython 3.12 exactly" >&2
  exit 2
}
command -v "${gcloud_bin}" >/dev/null 2>&1 || {
  echo "gcloud CLI is required on the controller" >&2
  exit 2
}

version_json="$("${gcloud_bin}" version --format=json)" || {
  echo "unable to read the gcloud SDK version" >&2
  exit 2
}
if ! sdk_version="$(
  SUNFISH_GCLOUD_VERSION_JSON="${version_json}" "${python_bin}" -c \
    'import json,os; p=json.loads(os.environ["SUNFISH_GCLOUD_VERSION_JSON"]); v=p.get("Google Cloud SDK"); assert isinstance(v,str); print(v)'
)"; then
  echo "gcloud version did not return valid JSON with a Google Cloud SDK version" >&2
  exit 2
fi
sdk_major="${sdk_version%%.*}"
[[ "${sdk_major}" =~ ^[0-9]+$ && "${sdk_major}" -ge 344 ]] || {
  echo "gcloud ${sdk_version} is too old; Sunfish requires Google Cloud SDK >= 344" >&2
  exit 2
}

SUNFISH_GCLOUD_BIN="${gcloud_bin}" "${script_dir}/tpu_iap.sh" check-cli
active_account="$(
  "${gcloud_bin}" auth list \
    --filter=status:ACTIVE \
    '--format=value(account)'
)" || {
  echo "unable to inspect the active gcloud account" >&2
  exit 2
}
[[ -n "${active_account}" && "${active_account}" != *$'\n'* ]] || {
  echo "exactly one active gcloud account is required" >&2
  exit 2
}

"${python_bin}" "${root}/scripts/check_tpu_release_safety.py"
if ((local_only)); then
  echo "Sunfish controller local preflight passed: gcloud=${sdk_version} account=${active_account}"
  exit 0
fi

: "${TPU_NAME:?set TPU_NAME to the allocation name}"
: "${PROJECT_ID:?set PROJECT_ID to the allocation project}"
: "${ZONE:?set ZONE to the allocation zone}"
[[ "${TPU_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$ ]] || {
  echo "TPU_NAME has an unsafe format" >&2
  exit 2
}
[[ "${PROJECT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9:._-]{1,127}$ ]] || {
  echo "PROJECT_ID has an unsafe format" >&2
  exit 2
}
[[ "${ZONE}" =~ ^[a-z][a-z0-9-]*-[a-z0-9]+-[a-z]$ ]] || {
  echo "ZONE has an unsafe format" >&2
  exit 2
}
configured_project="$("${gcloud_bin}" config get-value project --quiet)" || {
  echo "unable to inspect the configured gcloud project" >&2
  exit 2
}
[[ "${configured_project}" == "${PROJECT_ID}" ]] || {
  echo "active gcloud project ${configured_project:-unset} differs from PROJECT_ID=${PROJECT_ID}" >&2
  exit 2
}

for confirmation in \
  SUNFISH_IAP_TUNNEL_ROLE_CONFIRMED \
  SUNFISH_IAP_SSH_FIREWALL_CONFIRMED \
  SUNFISH_PRIVATE_GOOGLE_ACCESS_CONFIRMED \
  SUNFISH_GCS_IAM_CONFIRMED; do
  [[ "${!confirmation:-}" == 1 ]] || {
    echo "${confirmation}=1 is required after allocation-owner verification" >&2
    exit 2
  }
done

command -v "${ssh_add_bin}" >/dev/null 2>&1 || {
  echo "ssh-add is required for all-worker TPU SSH" >&2
  exit 2
}
ssh_private_key="${SUNFISH_COMPUTE_SSH_KEY:-${HOME}/.ssh/google_compute_engine}"
ssh_public_key="${ssh_private_key}.pub"
[[ -n "${SSH_AUTH_SOCK:-}" ]] || {
  echo "SSH_AUTH_SOCK is unset; start an ssh-agent and load the Compute Engine key" >&2
  exit 2
}
for key_path in "${ssh_private_key}" "${ssh_public_key}"; do
  [[ -f "${key_path}" && ! -L "${key_path}" ]] || {
    echo "missing regular Compute Engine SSH key file: ${key_path}" >&2
    exit 2
  }
done
if ! loaded_keys="$("${ssh_add_bin}" -L 2>/dev/null)"; then
  echo "ssh-agent has no readable identities; run ssh-add ${ssh_private_key}" >&2
  exit 2
fi
if ! SUNFISH_LOADED_SSH_KEYS="${loaded_keys}" "${python_bin}" -c \
  'import os,pathlib,sys; target=pathlib.Path(sys.argv[1]).read_text().split()[:2]; loaded=[line.split()[:2] for line in os.environ["SUNFISH_LOADED_SSH_KEYS"].splitlines()]; raise SystemExit(0 if len(target)==2 and target in loaded else 1)' \
  "${ssh_public_key}"; then
  echo "Compute Engine SSH key is not loaded; run ssh-add ${ssh_private_key}" >&2
  exit 2
fi

echo "Sunfish controller preflight passed without contacting TPU workers or GCS"
echo "gcloud=${sdk_version} account=${active_account} project=${PROJECT_ID} zone=${ZONE} tpu=${TPU_NAME}"
