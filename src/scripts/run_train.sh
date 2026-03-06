#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: run_train.sh <train_config.yaml>"
  exit 1
fi

python -m imm_qwen.train --config "$1"

