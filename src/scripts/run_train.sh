#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: run_train.sh <train_config.yaml> [accelerate_config.yaml]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

TRAIN_CONFIG="$1"
ACCELERATE_CONFIG="${2:-${SCRIPT_DIR}/accelerate_config.yaml}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" \
  -m imm_qwen.train --config "${TRAIN_CONFIG}"
