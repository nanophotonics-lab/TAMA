#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PREFIX="${MEEP_CONDA_PREFIX:-${CONDA_PREFIX:-}}"

if [[ -z "${CONDA_ENV_PREFIX}" ]]; then
  echo "Error: activate a conda environment or set MEEP_CONDA_PREFIX." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"

export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
export MEEP_CONDA_PREFIX="${CONDA_ENV_PREFIX}"

"${PYTHON_BIN}" -m pip install --no-build-isolation --editable "${SCRIPT_DIR}"
"${PYTHON_BIN}" -m pip install --no-build-isolation "${SCRIPT_DIR}/native/meep"
