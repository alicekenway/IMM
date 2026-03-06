from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .config import ImmQwenProjectConfig
from .controller import RuleBasedMemoryController
from .data_llamafactory import ImmDataCollator
from .memory_state import MultiScopeMemoryState
from .modeling_imm import QwenImmAdapter


@dataclass
class TrainBuildArtifacts:
    model: torch.nn.Module
    tokenizer: Any
    controller: RuleBasedMemoryController
    memory_state: MultiScopeMemoryState
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


def build_model_with_imm(project_config: ImmQwenProjectConfig) -> TrainBuildArtifacts:
    tokenizer = build_tokenizer(project_config)
    base_model = build_base_causal_lm_model(project_config)
    base_model.resize_token_embeddings(len(tokenizer))

    hidden_dim = int(base_model.config.hidden_size)
    controller = RuleBasedMemoryController(project_config.controller)
    memory_state = MultiScopeMemoryState(
        key_dim=project_config.memory_dimensions.key_dim,
        value_dim=project_config.memory_dimensions.value_dim,
        working_slots=project_config.memory_slots.working_slots,
        session_slots=project_config.memory_slots.session_slots,
    )

    wrapped_model = QwenImmAdapter(
        base_model=base_model,
        placement_config=project_config.placement,
        controller=controller,
        memory_state=memory_state,
        hidden_dim=hidden_dim,
        key_dim=project_config.memory_dimensions.key_dim,
        value_dim=project_config.memory_dimensions.value_dim,
        summary_config=project_config.turn_summary,
    )
    wrapped_model = attach_lora_to_qwen(wrapped_model, project_config)
    data_collator = ImmDataCollator(tokenizer)
    return TrainBuildArtifacts(
        model=wrapped_model,
        tokenizer=tokenizer,
        controller=controller,
        memory_state=memory_state,
        data_collator=data_collator,
    )


def resolve_imm_adapter(model: torch.nn.Module) -> QwenImmAdapter:
    if isinstance(model, QwenImmAdapter):
        return model
    if hasattr(model, "module"):
        module = getattr(model, "module")
        if isinstance(module, QwenImmAdapter):
            return module
    # Handle PEFT and similar wrappers.
    candidate_attrs = ("base_model", "model")
    for attr_name in candidate_attrs:
        if hasattr(model, attr_name):
            nested = getattr(model, attr_name)
            if isinstance(nested, QwenImmAdapter):
                return nested
            if isinstance(nested, torch.nn.Module):
                try:
                    return resolve_imm_adapter(nested)
                except ValueError:
                    pass
    raise ValueError("Cannot resolve QwenImmAdapter from the provided model.")


def prefill_history_memory(
    model: torch.nn.Module,
    history_input_ids: torch.Tensor,
    history_attention_mask: torch.Tensor,
    history_line_mask: torch.Tensor,
) -> None:
    """
    Prefill session memory from history lines before current-turn supervision.

    Tensor shapes:
      - history_input_ids: [B, H, T_hist]
      - history_attention_mask: [B, H, T_hist]
      - history_line_mask: [B, H]
    """
    adapter = resolve_imm_adapter(model)
    imm_module = adapter.get_last_imm_module()
    if imm_module is None:
        return

    batch_size, history_lines_count, _ = history_input_ids.shape
    if history_lines_count == 0:
        return

    device = next(model.parameters()).device
    history_input_ids = history_input_ids.to(device)
    history_attention_mask = history_attention_mask.to(device)
    history_line_mask = history_line_mask.to(device)

    for history_line_index in range(history_lines_count):
        active_rows = history_line_mask[:, history_line_index]
        if not torch.any(active_rows):
            continue

        line_input_ids = history_input_ids[:, history_line_index, :]
        line_attention_mask = history_attention_mask[:, history_line_index, :]
        # During history prefill we avoid recursive history lookups by masking
        # all positions from reading existing session memory.
        line_history_lookup_mask = torch.ones_like(line_attention_mask, dtype=torch.bool)
        adapter.set_history_lookup_mask(line_history_lookup_mask)

        outputs = model(
            input_ids=line_input_ids,
            attention_mask=line_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        final_hidden = outputs.hidden_states[-1]
        imm_module.write_session_summary(
            hidden_states=final_hidden,
            memory_state=adapter.memory_state,
            attention_mask=line_attention_mask,
            row_mask=active_rows,
        )


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


def build_loss_bundle(
    logits: torch.Tensor,
    labels: torch.Tensor,
    auxiliary_losses: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    vocab_size = logits.size(-1)
    lm_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
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

