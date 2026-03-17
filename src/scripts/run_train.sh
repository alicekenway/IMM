#!/usr/bin/env bash
set -euo pipefail

# Usage: run_train.sh <train_config.yaml> [options]
# Options:
#   --accelerate-config <path>    Path to accelerate config YAML (default: accelerate_config.yaml)
#   --conda-env <path>            Path to conda environment
#   --python <path>               Path to Python interpreter
#
# Examples:
#   bash run_train.sh ../../../examples/train_config.example.yaml
#   bash run_train.sh ../../../examples/train_config.example.yaml --conda-env /mnt/users/jinyang_wang/miniforge3/envs/nlu_expt
#   bash run_train.sh ../../../examples/train_config.example.yaml --accelerate-config accelerate_config_multi_gpu.yaml --conda-env /mnt/users/jinyang_wang/miniforge3/envs/nlu_expt

if [[ $# -lt 1 ]]; then
  echo "Usage: run_train.sh <train_config.yaml> [options]"
  echo ""
  echo "Options:"
  echo "  --accelerate-config <path>    Path to accelerate config YAML"
  echo "  --conda-env <path>            Path to conda environment"
  echo "  --python <path>               Path to Python interpreter"
  exit 1
fi

TRAIN_CONFIG="$1"
shift || true

ACCELERATE_CONFIG=""
CONDA_ENV=""
PYTHON_PATH=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerate-config)
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
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
  # Slurm non-interactive shells often do not preload `conda`.
  # Some conda activate hooks reference unset variables, so temporarily relax `set -u`.
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
      # Fallback for environments without a full conda initialization setup.
      source "${CONDA_ENV}/bin/activate"
    else
      echo "Error: could not initialize conda from: $CONDA_ENV"
      echo "Expected one of:"
      echo "  - $CONDA_SH"
      echo "  - ${CONDA_ENV}/bin/activate"
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

# Verify Python version
echo "Python version: $($PYTHON_CMD --version)"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"

# Resolve config paths
if [[ ! -f "$TRAIN_CONFIG" ]]; then
  echo "Error: Train config not found: $TRAIN_CONFIG"
  exit 1
fi
TRAIN_CONFIG="$(cd "$(dirname "$TRAIN_CONFIG")" && pwd)/$(basename "$TRAIN_CONFIG")"
echo "Train config: $TRAIN_CONFIG"

# Resolve accelerate config
if [[ -z "$ACCELERATE_CONFIG" ]]; then
  ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config.yaml"
else
  if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
    # Try relative to script directory
    if [[ -f "$SCRIPT_DIR/$ACCELERATE_CONFIG" ]]; then
      ACCELERATE_CONFIG="$SCRIPT_DIR/$ACCELERATE_CONFIG"
    else
      echo "Error: Accelerate config not found: $ACCELERATE_CONFIG"
      exit 1
    fi
  fi
  ACCELERATE_CONFIG="$(cd "$(dirname "$ACCELERATE_CONFIG")" && pwd)/$(basename "$ACCELERATE_CONFIG")"
fi
echo "Accelerate config: $ACCELERATE_CONFIG"

# Run training
echo "Starting training..."
accelerate launch --config_file "${ACCELERATE_CONFIG}" \
  -m imm_qwen.train --config "${TRAIN_CONFIG}"
