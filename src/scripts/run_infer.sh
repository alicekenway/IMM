#!/usr/bin/env bash
set -euo pipefail

# Usage: run_infer.sh <inference_config.yaml> [options]
# Options:
#   --conda-env <path>    Path to conda environment
#   --python <path>       Path to Python interpreter
#
# Examples:
#   bash run_infer.sh ../../../examples/inference_config.example.yaml
#   bash run_infer.sh ../../../examples/inference_config.example.yaml --conda-env /mnt/users/jinyang_wang/miniforge3/envs/nlu_expt

if [[ $# -lt 1 ]]; then
  echo "Usage: run_infer.sh <inference_config.yaml> [options]"
  echo ""
  echo "Options:"
  echo "  --conda-env <path>    Path to conda environment"
  echo "  --python <path>       Path to Python interpreter"
  exit 1
fi

INFER_CONFIG="$1"
shift || true

CONDA_ENV=""
PYTHON_PATH=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --python)
      PYTHON_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate conda environment if specified
if [[ -n "$CONDA_ENV" ]]; then
  echo "Activating conda environment: $CONDA_ENV"
  if [[ ! -d "$CONDA_ENV" ]]; then
    echo "Error: Conda environment not found at: $CONDA_ENV"
    exit 1
  fi
  _was_nounset=0
  if [[ "$-" == *u* ]]; then
    _was_nounset=1
    set +u
  fi

  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    CONDA_ROOT="$(cd "$(dirname "$CONDA_ENV")/.." && pwd)"
    CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
    if [[ -f "$CONDA_SH" ]]; then
      source "$CONDA_SH"
      conda activate "$CONDA_ENV"
    elif [[ -f "${CONDA_ENV}/bin/activate" ]]; then
      source "${CONDA_ENV}/bin/activate"
    else
      echo "Error: could not initialize conda from: $CONDA_ENV"
      exit 1
    fi
  fi
  if [[ $_was_nounset -eq 1 ]]; then
    set -u
  fi
  echo "Activated conda environment"
fi

# Use specified Python or default to current
if [[ -n "$PYTHON_PATH" ]]; then
  if [[ ! -f "$PYTHON_PATH" ]]; then
    echo "Error: Python interpreter not found at: $PYTHON_PATH"
    exit 1
  fi
  PYTHON_CMD="$PYTHON_PATH"
  echo "Using Python: $PYTHON_CMD"
else
  PYTHON_CMD="python"
  echo "Using default Python: $(which python)"
fi

echo "Python version: $($PYTHON_CMD --version)"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"

# Resolve config path
if [[ ! -f "$INFER_CONFIG" ]]; then
  echo "Error: Inference config not found: $INFER_CONFIG"
  exit 1
fi
INFER_CONFIG="$(cd "$(dirname "$INFER_CONFIG")" && pwd)/$(basename "$INFER_CONFIG")"
echo "Inference config: $INFER_CONFIG"

# Run inference
echo "Starting inference..."
$PYTHON_CMD -m imm_qwen.batch_infer --config "${INFER_CONFIG}"
