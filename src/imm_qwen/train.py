import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from .config import (
    DataSchemaConfig,
    ImmPlacementConfig,
    ImmQwenProjectConfig,
    LoraConfigSpec,
    MemoryControllerConfig,
    MemoryDimensionsConfig,
    MemorySlotsConfig,
    ModelBuildConfig,
    TrainingToolConfig,
    TurnSummaryConfig,
)
from .data_llamafactory import ImmSupervisedDataset
from .train_tools import (
    build_loss_bundle,
    build_model_with_imm,
    build_optimizer_groups,
    prefill_history_memory,
    resolve_imm_adapter,
)


def _load_yaml(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install with `pip install pyyaml`.") from exc
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _filter_known_fields(config_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    valid_keys = set(config_cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return {key: value for key, value in payload.items() if key in valid_keys}


def _build_project_config(payload: Dict[str, Any]) -> Tuple[ImmQwenProjectConfig, DataSchemaConfig, TrainingToolConfig]:
    model_config = ModelBuildConfig(**_filter_known_fields(ModelBuildConfig, payload["model"]))
    memory_dimensions = MemoryDimensionsConfig(
        **_filter_known_fields(MemoryDimensionsConfig, payload["memory_dimensions"])
    )

    memory_slots = MemorySlotsConfig(
        **_filter_known_fields(MemorySlotsConfig, payload.get("memory_slots", {}))
    )
    turn_summary = TurnSummaryConfig(
        **_filter_known_fields(TurnSummaryConfig, payload.get("turn_summary", {}))
    )
    controller = MemoryControllerConfig(
        **_filter_known_fields(MemoryControllerConfig, payload.get("controller", {}))
    )
    placement = ImmPlacementConfig(
        **_filter_known_fields(ImmPlacementConfig, payload.get("placement", {}))
    )
    lora = LoraConfigSpec(**_filter_known_fields(LoraConfigSpec, payload.get("lora", {})))

    data_payload = payload.get("data", {})
    if "dataset_path" not in data_payload:
        raise ValueError("config.data.dataset_path is required.")
    data_config = DataSchemaConfig(**_filter_known_fields(DataSchemaConfig, data_payload))

    training_config = TrainingToolConfig(
        **_filter_known_fields(TrainingToolConfig, payload.get("training", {}))
    )
    project_config = ImmQwenProjectConfig(
        model=model_config,
        memory_dimensions=memory_dimensions,
        memory_slots=memory_slots,
        turn_summary=turn_summary,
        controller=controller,
        placement=placement,
        lora=lora,
    )
    return project_config, data_config, training_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IMM-Qwen with YAML hyperparameters.")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = _load_yaml(args.config)
    project_config, data_config, training_config = _build_project_config(payload)
    _set_seed(training_config.seed)

    # IMM write parameters (summary_compressor, write_key_proj, write_value_proj)
    # are only used under torch.no_grad() during history prefill, so they never
    # receive gradients in the backward pass.  DDP would hang waiting for their
    # gradient all-reduce buckets without find_unused_parameters=True.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    artifacts = build_model_with_imm(project_config)

    dataset = ImmSupervisedDataset(
        tokenizer=artifacts.tokenizer,
        data_config=data_config,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=artifacts.data_collator,
    )

    optimizer_groups = build_optimizer_groups(
        model=artifacts.model,
        lora_lr=training_config.learning_rate_lora,
        imm_lr=training_config.learning_rate_imm,
        weight_decay=training_config.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_groups)

    model, optimizer, train_dataloader = accelerator.prepare(
        artifacts.model, optimizer, train_dataloader
    )

    model.train()
    global_step = 0

    for epoch_index in range(training_config.num_epochs):
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_dataloader):
            global_step += 1
            adapter = resolve_imm_adapter(model)
            # Efficiency mode: keep only turn-level session memory.
            # Working memory is reset every batch/turn.
            adapter.reset_session_memory()
            adapter.reset_working_memory()

            # 1) History lines are embedded into session memory first.
            with torch.no_grad():
                prefill_history_memory(
                    model=model,
                    history_input_ids=batch["history_input_ids"],
                    history_attention_mask=batch["history_attention_mask"],
                    history_line_mask=batch["history_line_mask"],
                )

            # 2) Present turn predicts output by querying session memory.
            # Mask convention:
            #   True  -> masked (do not read history at this token)
            #   False -> unmasked (allow history read at this token)
            adapter.set_history_lookup_mask(batch["history_lookup_mask"])

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                loss_bundle = build_loss_bundle(logits=outputs.logits, labels=batch["labels"])
                total_loss = loss_bundle["total_loss"]

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % training_config.log_every_steps == 0:
                loss_value = float(loss_bundle["total_loss"].detach().item())
                accelerator.print(
                    f"epoch={epoch_index} step={global_step} "
                    f"loss={loss_value:.6f} batch={batch_index}"
                )

        elapsed = time.time() - epoch_start
        accelerator.print(f"epoch={epoch_index} completed in {elapsed:.2f}s")

    accelerator.wait_for_everyone()

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        artifacts.tokenizer.save_pretrained(output_dir.as_posix())
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(output_dir.as_posix())

    accelerator.print("IMM-Qwen training completed.")
    accelerator.print(f"dataset_size={len(dataset)}")
    accelerator.print(f"optimizer_groups={len(optimizer_groups)} output_dir={output_dir.as_posix()}")
    final_adapter = resolve_imm_adapter(model)
    accelerator.print(f"selected_layers={getattr(final_adapter, 'selected_layer_indices', None)}")


if __name__ == "__main__":
    main()
