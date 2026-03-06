#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: run_infer.sh <config.json> <session_id> <text>"
  exit 1
fi

python -m imm_qwen.infer --config "$1" --session_id "$2" --text "$3"

