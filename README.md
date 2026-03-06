# IMM Qwen Project

This package implements an IMM-oriented trainer/runtime for Qwen 2/3 with:

- two memory scopes (`working`, `session`)
- mask-based history lookup routing
- turn-level compressed memory writes
- LoRA integration for Qwen modules
- trainer and inference tools

## File Purposes

- `imm_qwen/config.py`
  - Central config dataclasses for model/memory/data/training settings.
  - Keeps all hyperparameter naming consistent across train/infer tooling.

- `imm_qwen/interfaces.py`
  - Shared typed contracts (requests/results/protocols) for controller and memory.
  - Prevents hidden coupling between modules.

- `imm_qwen/memory_state.py`
  - Runtime memory bank implementation.
  - Pre-allocates per-batch tensors and performs vectorized read/write.
  - Implements FIFO replacement as baseline.

- `imm_qwen/controller.py`
  - Rule-based routing logic for memory usage.
  - Converts supervised labels into `history_lookup_mask` where:
    - `True` = masked, do not read history
    - `False` = unmasked, allowed to read history

- `imm_qwen/modeling_imm.py`
  - Core IMM algorithm:
    - query projection
    - memory retrieval
    - gated merge
    - turn-summary compression and memory write
  - Qwen block wrapper that injects IMM without forking HF internals.

- `imm_qwen/data_llamafactory.py`
  - Dataset adapter for LLaMA-Factory style records.
  - Splits each sample into:
    - history lines for memory prefill
    - present-turn prompt/output for supervised prediction

- `imm_qwen/train_tools.py`
  - Model/tokenizer builders.
  - LoRA attach utility.
  - Optimizer grouping.
  - History prefill routine that writes history lines into session memory.

- `imm_qwen/train.py`
  - End-to-end trainer entrypoint from YAML config.
  - Implements the required training logic in one loop.

- `imm_qwen/infer_tools.py`
  - Session lifecycle helpers for online inference usage.
  - Persist/load session memory states.

- `examples/train_config.example.yaml`
  - Example training hyperparameter file.
  - Includes model, data, memory, LoRA, and optimizer settings.

## Training Logic

For each sample, training follows:

1. Parse `history` into lines (if present).
2. Embed history lines first and write them into `session_memory`.
3. Build present-turn prompt (`system/instruction/input`) without history text replay.
4. Build `history_lookup_mask` from labels:
   - prompt tokens masked out from history lookup
   - output tokens allowed to read history
5. Use present turn to query history-memory and predict `output` with supervised LM loss.

This matches the required logic:
- history is embedded into memory
- current turn queries history-memory
- model predicts current `output`

## Train Command

Install dependencies first:

```bash
cd IMM
pip install -r requirements.txt
```

Use YAML for all hyperparameters:

```bash
cd IMM
PYTHONPATH=src python -m imm_qwen.train --config examples/train_config.example.yaml
```

Or with script:

```bash
cd IMM
bash src/scripts/run_train.sh examples/train_config.example.yaml
```

## Notes

- Requires installed packages from pip:
  - `transformers`
  - `peft`
  - `accelerate` (if `training.use_accelerate: true`)
  - `pyyaml`

