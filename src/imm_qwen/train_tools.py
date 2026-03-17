import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .config import ImmQwenProjectConfig
from .controller import RuleBasedMemoryController
from .data_llamafactory import ImmDataCollator
from .modeling_imm import QwenImmAdapter


@dataclass
class TrainBuildArtifacts:
    model: torch.nn.Module
    tokenizer: Any
    controller: RuleBasedMemoryController
    data_collator: ImmDataCollator


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype string: {dtype_name}")
    return mapping[key]


def build_tokenizer(project_config: ImmQwenProjectConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        project_config.model.model_name_or_path,
        trust_remote_code=project_config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_base_causal_lm_model(project_config: ImmQwenProjectConfig):
    from transformers import AutoModelForCausalLM

    torch_dtype = resolve_torch_dtype(project_config.model.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        project_config.model.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=project_config.model.trust_remote_code,
    )
    return model


def attach_lora_to_qwen(model: torch.nn.Module, project_config: ImmQwenProjectConfig) -> torch.nn.Module:
    if not project_config.lora.enabled:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for LoRA attachment. Install with `pip install peft`."
        ) from exc

    lora_config = LoraConfig(
        r=project_config.lora.rank,
        lora_alpha=project_config.lora.alpha,
        lora_dropout=project_config.lora.dropout,
        target_modules=list(project_config.lora.target_modules),
        bias=project_config.lora.bias,
        task_type=project_config.lora.task_type,
    )
    return get_peft_model(model, lora_config)


def mark_imm_parameters_trainable(model: torch.nn.Module) -> None:
    # PEFT LoRA freezes non-adapter parameters by default, but IMM modules
    # are part of the trainable path and must remain learnable.
    for name, parameter in model.named_parameters():
        if "imm_module" in name:
            parameter.requires_grad = True


def build_model_with_imm(project_config: ImmQwenProjectConfig) -> TrainBuildArtifacts:
    tokenizer = build_tokenizer(project_config)
    base_model = build_base_causal_lm_model(project_config)
    base_model.resize_token_embeddings(len(tokenizer))

    hidden_dim = int(base_model.config.hidden_size)
    controller = RuleBasedMemoryController(project_config.controller)

    wrapped_model = QwenImmAdapter(
        base_model=base_model,
        placement_config=project_config.placement,
        controller=controller,
        hidden_dim=hidden_dim,
        key_dim=project_config.memory_dimensions.key_dim,
        value_dim=project_config.memory_dimensions.value_dim,
        summary_config=project_config.turn_summary,
    )
    wrapped_model = attach_lora_to_qwen(wrapped_model, project_config)
    mark_imm_parameters_trainable(wrapped_model)
    data_collator = ImmDataCollator(tokenizer)
    return TrainBuildArtifacts(
        model=wrapped_model,
        tokenizer=tokenizer,
        controller=controller,
        data_collator=data_collator,
    )


def resolve_imm_adapter(model: torch.nn.Module) -> QwenImmAdapter:
    pending_models: List[torch.nn.Module] = [model]
    visited_model_ids = set()
    candidate_attrs = ("module", "base_model", "model")

    while pending_models:
        current_model = pending_models.pop()
        model_id = id(current_model)
        if model_id in visited_model_ids:
            continue
        visited_model_ids.add(model_id)

        if isinstance(current_model, QwenImmAdapter):
            return current_model

        for attr_name in candidate_attrs:
            try:
                nested_model = getattr(current_model, attr_name)
            except AttributeError:
                continue

            if isinstance(nested_model, QwenImmAdapter):
                return nested_model

            if not isinstance(nested_model, torch.nn.Module):
                continue
            if id(nested_model) == model_id:
                continue

            pending_models.append(nested_model)

    raise ValueError("Cannot resolve QwenImmAdapter from the provided model.")


def build_optimizer_groups(
    model: torch.nn.Module,
    lora_lr: float,
    imm_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    lora_params: List[torch.nn.Parameter] = []
    imm_params: List[torch.nn.Parameter] = []
    other_trainable_params: List[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(parameter)
        elif "imm_module" in name or "memory" in name or "controller" in name:
            imm_params.append(parameter)
        else:
            other_trainable_params.append(parameter)

    groups: List[Dict[str, Any]] = []
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    if imm_params:
        groups.append({"params": imm_params, "lr": imm_lr, "weight_decay": weight_decay})
    if other_trainable_params:
        groups.append(
            {"params": other_trainable_params, "lr": imm_lr, "weight_decay": weight_decay}
        )
    return groups


def _collect_imm_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract IMM module parameters from the model (works through DDP/PEFT wrappers)."""
    imm_adapter = resolve_imm_adapter(model)
    state: Dict[str, torch.Tensor] = {}
    for name, param in imm_adapter.named_parameters():
        if "imm_module" in name:
            state[name] = param.data.clone()
    return state


def _load_imm_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load IMM module parameters back into the model."""
    imm_adapter = resolve_imm_adapter(model)
    own_state = dict(imm_adapter.named_parameters())
    loaded = 0
    for name, saved_tensor in state_dict.items():
        if name in own_state:
            own_state[name].data.copy_(saved_tensor)
            loaded += 1
    if loaded == 0 and len(state_dict) > 0:
        raise ValueError(
            f"IMM state dict has {len(state_dict)} entries but none matched model parameters."
        )


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    epoch: int,
    global_step: int,
    training_config_dict: Dict[str, Any],
) -> None:
    """Save a full checkpoint: LoRA adapter, IMM weights, optimizer, training state."""
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    unwrapped = model
    # Unwrap DDP / accelerate wrappers
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    # Save LoRA adapter (via PEFT save_pretrained if available)
    lora_dir = ckpt_path / "lora_adapter"
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(lora_dir.as_posix())

    # Save IMM module weights separately
    imm_state = _collect_imm_state_dict(unwrapped)
    torch.save(imm_state, (ckpt_path / "imm_modules.pt").as_posix())

    # Save optimizer state
    torch.save(optimizer.state_dict(), (ckpt_path / "optimizer.pt").as_posix())

    # Save tokenizer
    tokenizer_dir = ckpt_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir.as_posix())

    # Save training state (for resume)
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        **training_config_dict,
    }
    (ckpt_path / "training_state.json").write_text(
        json.dumps(training_state, indent=2), encoding="utf-8"
    )


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load checkpoint weights and return the saved training state.

    Must be called BEFORE accelerator.prepare() for LoRA/IMM weights,
    but optimizer state is loaded AFTER accelerator.prepare() wraps the
    optimizer — so ``optimizer`` may be None on the first call and
    loaded separately via ``load_optimizer_state``.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    # Load LoRA adapter weights
    lora_dir = ckpt_path / "lora_adapter"
    if lora_dir.exists():
        try:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as load_safetensors

            adapter_file = lora_dir / "adapter_model.safetensors"
            if adapter_file.exists():
                lora_state = load_safetensors(adapter_file.as_posix())
            else:
                bin_file = lora_dir / "adapter_model.bin"
                lora_state = torch.load(bin_file.as_posix(), map_location="cpu", weights_only=True)
            set_peft_model_state_dict(unwrapped, lora_state)
        except ImportError:
            pass

    # Load IMM module weights
    imm_file = ckpt_path / "imm_modules.pt"
    if imm_file.exists():
        imm_state = torch.load(imm_file.as_posix(), map_location="cpu", weights_only=True)
        _load_imm_state_dict(unwrapped, imm_state)

    # Load optimizer if provided
    if optimizer is not None:
        opt_file = ckpt_path / "optimizer.pt"
        if opt_file.exists():
            optimizer.load_state_dict(
                torch.load(opt_file.as_posix(), map_location="cpu", weights_only=True)
            )

    # Load training state
    state_file = ckpt_path / "training_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text(encoding="utf-8"))
    return {}


def load_optimizer_state(checkpoint_dir: str, optimizer: torch.optim.Optimizer) -> None:
    """Load optimizer state dict from a checkpoint (call after accelerator.prepare)."""
    opt_file = Path(checkpoint_dir) / "optimizer.pt"
    if opt_file.exists():
        optimizer.load_state_dict(
            torch.load(opt_file.as_posix(), map_location="cpu", weights_only=True)
        )


def build_loss_bundle(
    logits: torch.Tensor,
    labels: torch.Tensor,
    auxiliary_losses: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    # Standard causal LM shift: logits[t] predicts token t+1.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    lm_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    total_loss = lm_loss
    loss_bundle: Dict[str, torch.Tensor] = {"lm_loss": lm_loss}
    if auxiliary_losses:
        for name, value in auxiliary_losses.items():
            loss_bundle[name] = value
            total_loss = total_loss + value
    loss_bundle["total_loss"] = total_loss
    return loss_bundle
