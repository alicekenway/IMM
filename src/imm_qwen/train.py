import argparse
import random
import time
from dataclasses import asdict
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
    load_checkpoint,
    load_optimizer_state,
    resolve_imm_adapter,
    save_checkpoint,
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


def _build_project_config(
    payload: Dict[str, Any],
) -> Tuple[ImmQwenProjectConfig, DataSchemaConfig, TrainingToolConfig]:
    model_config = ModelBuildConfig(
        **_filter_known_fields(ModelBuildConfig, payload["model"])
    )
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
    lora = LoraConfigSpec(
        **_filter_known_fields(LoraConfigSpec, payload.get("lora", {}))
    )

    data_payload = payload.get("data", {})
    if "dataset_path" not in data_payload:
        raise ValueError("config.data.dataset_path is required.")
    data_config = DataSchemaConfig(
        **_filter_known_fields(DataSchemaConfig, data_payload)
    )
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
    parser = argparse.ArgumentParser(
        description="Train IMM-Qwen with YAML hyperparameters."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config YAML."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from (overrides config).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = _load_yaml(args.config)
    project_config, data_config, training_config = _build_project_config(payload)
    _set_seed(training_config.seed)

    # Resolve checkpoint resume path (CLI --resume overrides config)
    resume_dir = args.resume or training_config.resume_from_checkpoint

    # All IMM parameters are used inside dual_stream_forward (called from
    # the DDP-tracked forward), so find_unused_parameters is not needed.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    artifacts = build_model_with_imm(project_config)

    # Load LoRA + IMM weights from checkpoint before accelerator.prepare
    resumed_state: Dict[str, Any] = {}
    if resume_dir is not None:
        accelerator.print(f"Resuming from checkpoint: {resume_dir}")
        resumed_state = load_checkpoint(
            checkpoint_dir=resume_dir,
            model=artifacts.model,
            optimizer=None,  # optimizer loaded after prepare
        )
        accelerator.print(
            f"Restored weights from step={resumed_state.get('global_step', '?')} "
            f"epoch={resumed_state.get('epoch', '?')}"
        )

    dataset = ImmSupervisedDataset(
        tokenizer=artifacts.tokenizer,
        data_config=data_config,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        persistent_workers=training_config.num_workers > 0,
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

    # Load optimizer state after accelerator.prepare (device mapping is ready)
    if resume_dir is not None:
        load_optimizer_state(resume_dir, optimizer)
        accelerator.print("Restored optimizer state.")

    # Determine starting epoch and step from checkpoint
    start_epoch = int(resumed_state.get("epoch", 0))
    global_step = int(resumed_state.get("global_step", 0))

    model.train()
    output_dir = Path(training_config.output_dir)

    # Serializable training config for checkpoint metadata
    training_config_dict = {
        "learning_rate_lora": training_config.learning_rate_lora,
        "learning_rate_imm": training_config.learning_rate_imm,
        "weight_decay": training_config.weight_decay,
        "batch_size": training_config.batch_size,
        "max_grad_norm": training_config.max_grad_norm,
        "grad_accum_steps": training_config.grad_accum_steps,
        "seed": training_config.seed,
    }

    accelerator.print(
        f"dataset_size={len(dataset)} "
        f"batches_per_epoch={len(train_dataloader)} "
        f"batch_size={training_config.batch_size} "
        f"num_workers={training_config.num_workers} "
        f"grad_accum_steps={training_config.grad_accum_steps} "
        f"start_epoch={start_epoch} start_step={global_step}"
    )

    for epoch_index in range(start_epoch, training_config.num_epochs):
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_dataloader):
            global_step += 1
            batch_fetch_elapsed = time.time() - epoch_start
            if batch_index == 0:
                accelerator.print(
                    f"epoch={epoch_index} batch={batch_index} "
                    f"batch_fetched_after={batch_fetch_elapsed:.2f}s "
                    f"input_shape={tuple(batch['input_ids'].shape)} "
                    f"history_shape={tuple(batch['history_input_ids'].shape)}"
                )

            with accelerator.accumulate(model):
                forward_start = time.time()
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    history_input_ids=batch["history_input_ids"],
                    history_attention_mask=batch["history_attention_mask"],
                    history_line_mask=batch["history_line_mask"],
                    history_lookup_mask=batch["history_lookup_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                forward_elapsed = time.time() - forward_start
                loss_bundle = build_loss_bundle(
                    logits=outputs.logits, labels=batch["labels"]
                )
                total_loss = loss_bundle["total_loss"]

                backward_start = time.time()
                accelerator.backward(total_loss)
                backward_elapsed = time.time() - backward_start

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), training_config.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_elapsed = time.time() - forward_start

            if batch_index == 0:
                accelerator.print(
                    f"epoch={epoch_index} batch={batch_index} "
                    f"forward_time={forward_elapsed:.2f}s "
                    f"backward_time={backward_elapsed:.2f}s "
                    f"step_time={step_elapsed:.2f}s"
                )

            if global_step % training_config.log_every_steps == 0:
                loss_value = float(loss_bundle["total_loss"].detach().item())
                accelerator.print(
                    f"epoch={epoch_index} step={global_step} "
                    f"loss={loss_value:.6f} batch={batch_index}"
                )

            # Periodic checkpoint saving
            if (
                training_config.save_every_steps > 0
                and global_step % training_config.save_every_steps == 0
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint_{global_step}"
                    save_checkpoint(
                        checkpoint_dir=ckpt_dir.as_posix(),
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        tokenizer=artifacts.tokenizer,
                        epoch=epoch_index,
                        global_step=global_step,
                        training_config_dict=training_config_dict,
                    )
                    accelerator.print(f"Saved checkpoint: {ckpt_dir}")

        elapsed = time.time() - epoch_start
        accelerator.print(f"epoch={epoch_index} completed in {elapsed:.2f}s")

    # --- save final model ---------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        save_checkpoint(
            checkpoint_dir=final_dir.as_posix(),
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            tokenizer=artifacts.tokenizer,
            epoch=training_config.num_epochs,
            global_step=global_step,
            training_config_dict=training_config_dict,
        )
        accelerator.print(f"Saved final model: {final_dir}")

    accelerator.print("IMM-Qwen training completed.")
    accelerator.print(f"dataset_size={len(dataset)}")
    accelerator.print(
        f"optimizer_groups={len(optimizer_groups)} "
        f"output_dir={output_dir.as_posix()}"
    )
    final_adapter = resolve_imm_adapter(model)
    accelerator.print(
        f"selected_layers={getattr(final_adapter, 'selected_layer_indices', None)}"
    )


if __name__ == "__main__":
    main()
