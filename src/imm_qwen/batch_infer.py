"""Inference for IMM-Qwen.

Reads a JSON dataset (same format as training data), runs inference on each
sample with optional history prefill via the IMM memory mechanism, and writes
an output JSON that mirrors the input with an added ``inference_output`` field.

Usage:
    python -m imm_qwen.batch_infer --config inference_config.yaml
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

from .config import (
    ImmPlacementConfig,
    ImmQwenProjectConfig,
    LoraConfigSpec,
    MemoryControllerConfig,
    MemoryDimensionsConfig,
    MemorySlotsConfig,
    ModelBuildConfig,
    TurnSummaryConfig,
)
from .data_llamafactory import (
    SupervisedRecord,
    build_present_turn_prompt_text,
    extract_history_lines,
)
from .modeling_imm import QwenImmAdapter
from .train_tools import (
    build_model_with_imm,
    load_checkpoint,
    resolve_imm_adapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference-specific config
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Controls how text is generated."""
    strategy: str = "greedy"            # "greedy" | "beam" | "sample"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    num_beams: int = 4                  # only used when strategy == "beam"
    repetition_penalty: float = 1.0
    max_new_tokens: int = 128
    length_penalty: float = 1.0         # beam search length penalty
    no_repeat_ngram_size: int = 0


@dataclass
class InferConfig:
    """Top-level config for inference, parsed from YAML."""
    # Model
    use_imm: bool = True                # False = ignore history prefill and run single-turn inference with the same trained model
    model_name_or_path: str = ""
    checkpoint_dir: str = ""            # dir saved by train (contains lora_adapter/, imm_modules.pt, tokenizer/)
    torch_dtype: str = "bf16"
    trust_remote_code: bool = False
    device: str = "auto"                # "auto" | "cuda" | "cuda:0" | "cpu"

    # IMM architecture (must match training config)
    memory_dimensions: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 896, "key_dim": 192, "value_dim": 192,
    })
    memory_slots: Dict[str, Any] = field(default_factory=lambda: {
        "session_slots": 64, "working_slots": 16,
    })
    turn_summary: Dict[str, Any] = field(default_factory=lambda: {
        "pooling_strategy": "last_token", "use_layer_norm": True,
    })
    controller: Dict[str, Any] = field(default_factory=lambda: {
        "session_merge_gate": 1.0, "use_working_memory": False, "working_merge_gate": 1.0,
    })
    placement: Dict[str, Any] = field(default_factory=lambda: {
        "enable_imm": True, "top_fraction": 0.5,
    })
    lora: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none", "task_type": "CAUSAL_LM",
    })

    # Data
    data_path: str = ""
    output_path: str = "inference_output.json"
    max_history_line_length: int = 256

    # Generation
    generation: Dict[str, Any] = field(default_factory=dict)

    # Misc
    seed: int = 42
    log_every: int = 10


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _filter_known_fields(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    valid = set(cls.__dataclass_fields__.keys())
    return {k: v for k, v in payload.items() if k in valid}


def load_infer_config(yaml_path: str) -> Tuple[InferConfig, GenerationConfig]:
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))

    defaults = InferConfig()
    cfg = InferConfig(
        use_imm=raw.get("use_imm", defaults.use_imm),
        model_name_or_path=raw.get("model_name_or_path", defaults.model_name_or_path),
        checkpoint_dir=raw.get("checkpoint_dir", defaults.checkpoint_dir),
        torch_dtype=raw.get("torch_dtype", defaults.torch_dtype),
        trust_remote_code=raw.get("trust_remote_code", defaults.trust_remote_code),
        device=raw.get("device", defaults.device),
        memory_dimensions=raw.get("memory_dimensions", defaults.memory_dimensions),
        memory_slots=raw.get("memory_slots", defaults.memory_slots),
        turn_summary=raw.get("turn_summary", defaults.turn_summary),
        controller=raw.get("controller", defaults.controller),
        placement=raw.get("placement", defaults.placement),
        lora=raw.get("lora", defaults.lora),
        data_path=raw.get("data_path", defaults.data_path),
        output_path=raw.get("output_path", defaults.output_path),
        max_history_line_length=raw.get("max_history_line_length", defaults.max_history_line_length),
        generation=raw.get("generation", {}),
        seed=raw.get("seed", defaults.seed),
        log_every=raw.get("log_every", defaults.log_every),
    )

    gen_raw = raw.get("generation", {})
    gen = GenerationConfig(**_filter_known_fields(GenerationConfig, gen_raw))

    return cfg, gen


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _build_project_config(cfg: InferConfig) -> ImmQwenProjectConfig:
    """Reconstruct the ImmQwenProjectConfig needed by build_model_with_imm."""
    model_config = ModelBuildConfig(
        model_name_or_path=cfg.model_name_or_path,
        torch_dtype=cfg.torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    mem_dim = MemoryDimensionsConfig(**_filter_known_fields(MemoryDimensionsConfig, cfg.memory_dimensions))
    mem_slots = MemorySlotsConfig(**_filter_known_fields(MemorySlotsConfig, cfg.memory_slots))
    turn_summary = TurnSummaryConfig(**_filter_known_fields(TurnSummaryConfig, cfg.turn_summary))
    controller = MemoryControllerConfig(**_filter_known_fields(MemoryControllerConfig, cfg.controller))
    placement = ImmPlacementConfig(**_filter_known_fields(ImmPlacementConfig, cfg.placement))

    lora_payload = dict(cfg.lora)
    if "target_modules" in lora_payload and isinstance(lora_payload["target_modules"], list):
        lora_payload["target_modules"] = tuple(lora_payload["target_modules"])
    # Force dropout to 0 during inference
    lora_payload["dropout"] = 0.0
    lora_spec = LoraConfigSpec(**_filter_known_fields(LoraConfigSpec, lora_payload))

    return ImmQwenProjectConfig(
        model=model_config,
        memory_dimensions=mem_dim,
        memory_slots=mem_slots,
        turn_summary=turn_summary,
        controller=controller,
        placement=placement,
        lora=lora_spec,
    )


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_generation_kwargs(gen: GenerationConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": gen.max_new_tokens,
        "repetition_penalty": gen.repetition_penalty,
    }
    if gen.no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = gen.no_repeat_ngram_size

    if gen.strategy == "greedy":
        kwargs["do_sample"] = False
        kwargs["num_beams"] = 1
    elif gen.strategy == "beam":
        kwargs["do_sample"] = False
        kwargs["num_beams"] = gen.num_beams
        kwargs["length_penalty"] = gen.length_penalty
        kwargs["early_stopping"] = True
    elif gen.strategy == "sample":
        kwargs["do_sample"] = True
        kwargs["temperature"] = gen.temperature
        kwargs["top_p"] = gen.top_p
        kwargs["top_k"] = gen.top_k
    else:
        raise ValueError(f"Unknown generation strategy: {gen.strategy!r}. Use 'greedy', 'beam', or 'sample'.")

    return kwargs


# ---------------------------------------------------------------------------
# History prefill
# ---------------------------------------------------------------------------

def prefill_history(
    model: torch.nn.Module,
    adapter: QwenImmAdapter,
    tokenizer: Any,
    history_lines: List[str],
    device: torch.device,
    max_line_length: int = 256,
) -> None:
    """Run each history line through the model in history_collect mode to
    populate the per-layer memory banks."""
    if not history_lines:
        return

    for line_text in history_lines:
        encoded = tokenizer(
            line_text,
            truncation=True,
            max_length=max_line_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        adapter.set_history_collect_mode(attention_mask=attention_mask)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    adapter.set_passthrough_mode()


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def run_single(
    model: torch.nn.Module,
    adapter: QwenImmAdapter,
    tokenizer: Any,
    record: SupervisedRecord,
    gen_kwargs: Dict[str, Any],
    device: torch.device,
    max_history_line_length: int = 256,
) -> str:
    """Run inference on one sample, handling history prefill and memory."""
    # 1. Clear memory from previous sample
    adapter.clear_all_memory_slots()

    # 2. If the sample has history, prefill it into the memory banks
    history_lines = extract_history_lines(record.history)
    has_history = len(history_lines) > 0

    if has_history:
        prefill_history(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            history_lines=history_lines,
            device=device,
            max_line_length=max_history_line_length,
        )

    # 3. Build prompt (same format as training)
    prompt_text = build_present_turn_prompt_text(record)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = input_ids.size(1)

    # 4. Set mode: if history was loaded, use present_query mode so generated
    #    tokens can read from the memory banks. Otherwise passthrough is fine.
    if has_history:
        adapter.set_present_query_mode(
            history_lookup_mask=None,
            prompt_length=prompt_length,
        )
    else:
        adapter.set_passthrough_mode()

    # 5. Generate
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            **gen_kwargs,
        )

    # 6. Decode only the newly generated tokens
    generated_tokens = output_ids[0, prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 7. Reset mode
    adapter.set_passthrough_mode()
    adapter.clear_all_memory_slots()

    return response


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(yaml_path: str) -> None:
    cfg, gen = load_infer_config(yaml_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Seed
    torch.manual_seed(cfg.seed)

    # Load the checkpoint tokenizer *before* building the model so that
    # resize_token_embeddings() uses the correct vocab size. The checkpoint
    # tokenizer may have added special tokens that differ from the base model.
    project_config = _build_project_config(cfg)
    ckpt_path = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    ckpt_tokenizer_dir = ckpt_path / "tokenizer" if ckpt_path else None
    checkpoint_tokenizer = None

    if ckpt_tokenizer_dir and ckpt_tokenizer_dir.exists():
        from transformers import AutoTokenizer
        logger.info("Loading checkpoint tokenizer: %s", ckpt_tokenizer_dir)
        checkpoint_tokenizer = AutoTokenizer.from_pretrained(
            ckpt_tokenizer_dir.as_posix(),
            trust_remote_code=project_config.model.trust_remote_code,
        )
        if checkpoint_tokenizer.pad_token is None and checkpoint_tokenizer.eos_token is not None:
            checkpoint_tokenizer.pad_token = checkpoint_tokenizer.eos_token

    # Build model with IMM + LoRA (same architecture as training).
    logger.info("Building model from base: %s", cfg.model_name_or_path)
    artifacts = build_model_with_imm(project_config)
    model = artifacts.model

    # Use the checkpoint tokenizer if available, and re-resize embeddings
    # to match its vocab (overriding the base-tokenizer resize done inside
    # build_model_with_imm).
    if checkpoint_tokenizer is not None:
        tokenizer = checkpoint_tokenizer
        resolve_imm_adapter(model).original_model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = artifacts.tokenizer

    # Load trained checkpoint weights (LoRA + IMM)
    if ckpt_path and ckpt_path.exists():
        logger.info("Loading checkpoint from: %s", cfg.checkpoint_dir)
        load_checkpoint(cfg.checkpoint_dir, model)
    elif cfg.checkpoint_dir:
        logger.warning("Checkpoint dir not found: %s — running with base weights", cfg.checkpoint_dir)

    # Move to device
    device = _resolve_device(cfg.device)
    model = model.to(device)
    model.eval()
    adapter = resolve_imm_adapter(model)

    if not cfg.use_imm:
        logger.info("use_imm=False — history will be ignored (single-turn inference)")

    # Build generation kwargs
    gen_kwargs = build_generation_kwargs(gen)
    logger.info("Generation config: strategy=%s %s", gen.strategy, gen_kwargs)

    # Load input data
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {cfg.data_path}")
    raw_data: List[Dict[str, Any]] = json.loads(data_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d samples from %s", len(raw_data), cfg.data_path)

    # Run inference sample by sample
    results: List[Dict[str, Any]] = []
    total = len(raw_data)
    start_time = time.time()

    for idx, row in enumerate(raw_data):
        record = SupervisedRecord(
            instruction=str(row.get("instruction", "")),
            input=str(row.get("input", "")),
            output=str(row.get("output", "")),
            system=str(row.get("system", "")),
            history=row.get("history") if cfg.use_imm else None,
        )

        response = run_single(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            record=record,
            gen_kwargs=gen_kwargs,
            device=device,
            max_history_line_length=cfg.max_history_line_length,
        )

        # Copy all original fields and add inference_output
        result = dict(row)
        result["inference_output"] = response
        results.append(result)

        if cfg.log_every > 0 and (idx + 1) % cfg.log_every == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            logger.info(
                "[%d/%d] %.1f samples/s | input: %s | output: %s",
                idx + 1, total, speed,
                record.input[:60], response[:60],
            )

    # Write output
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    elapsed = time.time() - start_time
    logger.info(
        "Done. %d samples in %.1fs (%.2f samples/s). Output: %s",
        total, elapsed, total / elapsed if elapsed > 0 else 0, cfg.output_path,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IMM-Qwen inference.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to inference YAML config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()
