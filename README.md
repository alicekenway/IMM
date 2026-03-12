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

## Model Structure (How IMM is wired into Qwen)

This project keeps the Hugging Face Qwen model mostly intact and injects IMM by **wrapping selected decoder layers in place**.

- **High-level components**
  - **Base model**: a standard `AutoModelForCausalLM` (Qwen).
  - **`QwenImmAdapter`** (`imm_qwen/modeling_imm.py`): owns the base model and replaces some decoder layers with IMM-enabled wrappers.
  - **`QwenImmLayerWrapper`**: wraps one decoder layer; runs the original layer first, then applies IMM to its output hidden states.
  - **`ImplicitMemoryModule` (IMM core)**: does “read → gated merge” during forward; does “turn summary → write” when requested.
  - **`MultiScopeMemoryState`** (`imm_qwen/memory_state.py`): two memory banks per sample:
    - **`session`**: long-term turn-level memory (history/past turns)
    - **`working`**: short-term scratch memory (optional; often disabled for efficiency)
  - **`RuleBasedMemoryController`** (`imm_qwen/controller.py`): applies deterministic gates and enforces the `history_lookup_mask` rule.

- **Layer placement**
  - Controlled by `placement` config.
  - If `placement.enable_imm: false`, no layers are wrapped.
  - If `selected_layer_indices` is not provided, the adapter wraps the **top `top_fraction`** of decoder layers (e.g. 0.5 → top half).

- **Forward pass: what happens inside a wrapped layer**

  Conceptually, a wrapped layer computes:

  1. Run the original Qwen decoder layer to get `hidden_states` \([B, T, H]\).
  2. IMM computes a query per token: `query = query_proj(hidden_states)` \([B, T, key_dim]\).
  3. IMM reads memory:
     - `session` read is conditioned by `history_lookup_mask`.
     - `working` read happens only if `controller.use_working_memory: true`.
  4. Controller applies **gated merge**:
     - A scalar gate (`session_merge_gate` / `working_merge_gate`) scales retrieved vectors.
     - `history_lookup_mask` blocks history usage at masked tokens:
       - `True`  ⇒ masked ⇒ do **not** merge retrieved memory at that token
       - `False` ⇒ allowed ⇒ merge retrieved memory at that token
  5. Retrieved values are projected back to hidden size and added as a residual:
     - `hidden_out = LayerNorm(hidden_states + output_proj(working + session))`

  You can think of the per-layer structure like this:

  ```text
  tokens → [Qwen decoder layer] → hidden_states
                                 │
                                 ├─ IMM read: query_proj → memory_state.read(session/working)
                                 ├─ IMM gate: controller.merge_gate (+ history_lookup_mask)
                                 └─ IMM merge: output_proj + residual + LayerNorm → hidden_out
  ```

- **Memory write: how a “turn summary” is created**
  - IMM writes are turn-level (one vector per sequence / per history line), not per-token.
  - The write path compresses \([B, T, H]\) into \([B, value_dim]\) using `turn_summary.pooling_strategy`:
    - `last_token`: take the last non-padding token hidden state (via `attention_mask`)
    - `mean_pool`: mean over unmasked tokens
    - `attention_pool`: learned attention weights over tokens
  - The summary is optionally normalized (`turn_summary.use_layer_norm`), then projected into:
    - **write key** (`write_key_proj(summary)`) for retrieval addressing
    - **write value** (`write_value_proj(summary)`) for what will be retrieved/merged later

- **Where writes happen in this repo**
  - **Training** (`imm_qwen/train_tools.py` + `imm_qwen/train.py`):
    - History is split into lines.
    - Each history line is forwarded once and written into **session memory** as a summary (“prefill”).
    - Then the present turn is trained with LM loss while being allowed to read session memory only on target tokens (via `history_lookup_mask`).
  - **Inference** (`imm_qwen/infer_tools.py`):
    - After producing an assistant response, the current turn is summarized and written into **session memory** so later turns can retrieve it.

