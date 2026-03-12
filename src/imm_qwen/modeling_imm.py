from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import ImmPlacementConfig, TurnSummaryConfig
from .controller import RuleBasedMemoryController
from .interfaces import MemoryMetadata, MemoryReadRequest, MemoryWriteRequest
from .memory_state import MultiScopeMemoryState


@dataclass(frozen=True)
class ImmForwardStats:
    session_attention: Optional[torch.Tensor]
    working_attention: Optional[torch.Tensor]


class TurnSummaryCompressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        value_dim: int,
        summary_config: Optional[TurnSummaryConfig] = None,
    ) -> None:
        super().__init__()
        self.summary_config = summary_config or TurnSummaryConfig()
        self.summary_proj = nn.Linear(hidden_dim, value_dim, bias=False)
        self.summary_pooling_logits = nn.Linear(hidden_dim, 1, bias=False)
        self.output_norm = nn.LayerNorm(value_dim) if self.summary_config.use_layer_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compress a full turn into one latent summary vector per sample.
        if self.summary_config.pooling_strategy == "last_token":
            pooled = self._last_token_pool(hidden_states, attention_mask)
        elif self.summary_config.pooling_strategy == "mean_pool":
            pooled = self._mean_pool(hidden_states, attention_mask)
        elif self.summary_config.pooling_strategy == "attention_pool":
            pooled = self._attention_pool(hidden_states, attention_mask)
        else:
            raise ValueError(f"unsupported pooling strategy: {self.summary_config.pooling_strategy}")

        summary = self.summary_proj(pooled)
        return self.output_norm(summary)

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        lengths = attention_mask.to(torch.long).sum(dim=1).clamp(min=1)
        last_indices = lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _attention_pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        scores = self.summary_pooling_logits(hidden_states).squeeze(-1)  # [B, T]
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", weights, hidden_states)


class ImplicitMemoryModule(nn.Module):
    """
    IMM read/merge module for a transformer hidden-state stream.

    This module focuses on vectorized operations:
      - memory read by einsum
      - gated merge in batch
      - write performed at turn summary boundaries via dedicated method
    """

    def __init__(
        self,
        hidden_dim: int,
        key_dim: int,
        value_dim: int,
        controller: RuleBasedMemoryController,
        summary_config: Optional[TurnSummaryConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.controller = controller

        self.query_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.output_proj = nn.Linear(value_dim, hidden_dim, bias=False)
        self.write_key_proj = nn.Linear(value_dim, key_dim, bias=False)
        self.write_value_proj = nn.Linear(value_dim, value_dim, bias=False)
        self.merge_norm = nn.LayerNorm(hidden_dim)
        self.summary_compressor = TurnSummaryCompressor(
            hidden_dim=hidden_dim,
            value_dim=value_dim,
            summary_config=summary_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: MultiScopeMemoryState,
        history_lookup_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ImmForwardStats]:
        # Ensure bank tensors are allocated once per batch shape/device.
        batch_size = hidden_states.size(0)
        memory_state.ensure_batch_size(
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        query = self.query_proj(hidden_states)

        # Efficiency mode: skip working-memory read when disabled.
        if self.controller.config.use_working_memory:
            working_result = memory_state.read(
                MemoryReadRequest(
                    query=query,
                    scope="working",
                    history_lookup_mask=None,
                )
            )
            working_retrieved = working_result.retrieved
            working_attention = working_result.attention_weights
        else:
            working_retrieved = torch.zeros(
                hidden_states.size(0),
                hidden_states.size(1),
                self.value_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            working_attention = None

        session_result = memory_state.read(
            MemoryReadRequest(
                query=query,
                scope="session",
                history_lookup_mask=history_lookup_mask,
            )
        )

        # Apply controller gates. The mask convention is:
        #   True  -> masked out (no history lookup merge)
        #   False -> allowed to merge retrieved memory.
        working_gated = self.controller.merge_gate(
            hidden_states=hidden_states,
            retrieved_states=working_retrieved,
            scope="working",
            history_lookup_mask=history_lookup_mask,
        )
        session_gated = self.controller.merge_gate(
            hidden_states=hidden_states,
            retrieved_states=session_result.retrieved,
            scope="session",
            history_lookup_mask=history_lookup_mask,
        )

        merged = self.output_proj(working_gated + session_gated)
        hidden_out = self.merge_norm(hidden_states + merged)
        stats = ImmForwardStats(
            session_attention=session_result.attention_weights,
            working_attention=working_attention,
        )
        return hidden_out, stats

    def write_session_summary(
        self,
        hidden_states: torch.Tensor,
        memory_state: MultiScopeMemoryState,
        attention_mask: Optional[torch.Tensor] = None,
        turn_index: Optional[int] = None,
        retention_score: Optional[float] = None,
        row_mask: Optional[torch.Tensor] = None,
    ) -> None:
        summary = self.summary_compressor(hidden_states, attention_mask)
        key = self.write_key_proj(summary)
        value = self.write_value_proj(summary)
        memory_state.write(
            MemoryWriteRequest(
                key=key,
                value=value,
                scope="session",
                metadata=MemoryMetadata(turn_index=turn_index, retention_score=retention_score),
                row_mask=row_mask,
            )
        )

    def write_working_summary(
        self,
        hidden_states: torch.Tensor,
        memory_state: MultiScopeMemoryState,
        attention_mask: Optional[torch.Tensor] = None,
        row_mask: Optional[torch.Tensor] = None,
    ) -> None:
        summary = self.summary_compressor(hidden_states, attention_mask)
        key = self.write_key_proj(summary)
        value = self.write_value_proj(summary)
        memory_state.write(
            MemoryWriteRequest(
                key=key,
                value=value,
                scope="working",
                metadata=None,
                row_mask=row_mask,
            )
        )


class QwenImmLayerWrapper(nn.Module):
    """
    Wrap one decoder layer and apply IMM after the base layer output.
    """

    def __init__(self, base_layer: nn.Module, imm_module: ImplicitMemoryModule) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.imm_module = imm_module
        self.memory_state: Optional[MultiScopeMemoryState] = None
        # True means masked out / no history lookup for that token.
        self.history_lookup_mask: Optional[torch.Tensor] = None

    def set_memory_state(self, memory_state: Optional[MultiScopeMemoryState]) -> None:
        self.memory_state = memory_state

    def set_history_lookup_mask(self, history_lookup_mask: Optional[torch.Tensor]) -> None:
        self.history_lookup_mask = history_lookup_mask

    def forward(self, *args, **kwargs):
        # Run original decoder layer first, then inject IMM on hidden states.
        base_output = self.base_layer(*args, **kwargs)
        if self.memory_state is None:
            return base_output

        if isinstance(base_output, tuple):
            hidden_states = base_output[0]
            hidden_states, _ = self.imm_module(
                hidden_states=hidden_states,
                memory_state=self.memory_state,
                history_lookup_mask=self.history_lookup_mask,
            )
            return (hidden_states, *base_output[1:])

        if torch.is_tensor(base_output):
            hidden_states, _ = self.imm_module(
                hidden_states=base_output,
                memory_state=self.memory_state,
                history_lookup_mask=self.history_lookup_mask,
            )
            return hidden_states

        raise TypeError(
            "Unsupported decoder layer output type. Expected tensor or tuple with hidden states first."
        )


class QwenImmAdapter(nn.Module):
    """
    Adapter that injects IMM wrappers into selected decoder layers of a base model.

    It avoids forking transformers internals by replacing layer modules in place.
    """

    def __init__(
        self,
        base_model: nn.Module,
        placement_config: ImmPlacementConfig,
        controller: RuleBasedMemoryController,
        memory_state: MultiScopeMemoryState,
        hidden_dim: int,
        key_dim: int,
        value_dim: int,
        summary_config: Optional[TurnSummaryConfig] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.placement_config = placement_config
        self.controller = controller
        self.memory_state = memory_state
        self.summary_config = summary_config

        # Locate decoder layers from common HF model layouts and wrap selected
        # layers in place. This keeps compatibility with upstream transformers.
        layers = self._locate_decoder_layers(base_model)
        selected_indices = _resolve_selected_layer_indices(
            total_layers=len(layers),
            placement_config=placement_config,
        )

        self.wrapped_layers: List[QwenImmLayerWrapper] = []
        for layer_index in selected_indices:
            imm_module = ImplicitMemoryModule(
                hidden_dim=hidden_dim,
                key_dim=key_dim,
                value_dim=value_dim,
                controller=controller,
                summary_config=summary_config,
            )
            wrapper = QwenImmLayerWrapper(layers[layer_index], imm_module=imm_module)
            wrapper.set_memory_state(memory_state)
            layers[layer_index] = wrapper
            self.wrapped_layers.append(wrapper)

        self.selected_layer_indices = tuple(selected_indices)

    def set_memory_state(self, memory_state: MultiScopeMemoryState) -> None:
        self.memory_state = memory_state
        for wrapped_layer in self.wrapped_layers:
            wrapped_layer.set_memory_state(memory_state)

    def set_history_lookup_mask(self, history_lookup_mask: Optional[torch.Tensor]) -> None:
        for wrapped_layer in self.wrapped_layers:
            wrapped_layer.set_history_lookup_mask(history_lookup_mask)

    def reset_working_memory(self) -> None:
        self.memory_state.reset_working()

    def reset_session_memory(self) -> None:
        self.memory_state.reset_session()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Delegate generation input preparation to the wrapped base causal LM.

        PEFT's causal LM wrappers require this method to exist on the incoming
        model object when building LoRA adapters.
        """
        if not hasattr(self.base_model, "prepare_inputs_for_generation"):
            raise AttributeError("base_model does not provide prepare_inputs_for_generation().")
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        """
        Keep compatibility with generation internals used by some PEFT paths.
        """
        if not hasattr(self.base_model, "_prepare_encoder_decoder_kwargs_for_generation"):
            raise AttributeError(
                "base_model does not provide _prepare_encoder_decoder_kwargs_for_generation()."
            )
        return self.base_model._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if not hasattr(self.base_model, "generate"):
            raise AttributeError("base_model does not provide generate().")
        return self.base_model.generate(*args, **kwargs)

    def get_last_imm_module(self) -> Optional[ImplicitMemoryModule]:
        if not self.wrapped_layers:
            return None
        return self.wrapped_layers[-1].imm_module

    @staticmethod
    def _locate_decoder_layers(model: nn.Module) -> nn.ModuleList:
        candidates: List[Tuple[str, nn.ModuleList]] = []
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            candidates.append(("model.layers", model.model.layers))
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            candidates.append(("transformer.h", model.transformer.h))
        if hasattr(model, "layers"):
            maybe_layers = getattr(model, "layers")
            if isinstance(maybe_layers, nn.ModuleList):
                candidates.append(("layers", maybe_layers))

        if not candidates:
            raise ValueError(
                "Could not locate decoder layers. Expected one of: model.layers, transformer.h, layers."
            )
        return candidates[0][1]


def _resolve_selected_layer_indices(
    total_layers: int,
    placement_config: ImmPlacementConfig,
) -> List[int]:
    if not placement_config.enable_imm:
        return []

    if placement_config.selected_layer_indices is not None:
        resolved = []
        for idx in placement_config.selected_layer_indices:
            if idx < 0 or idx >= total_layers:
                raise ValueError(f"selected layer index out of range: {idx}")
            resolved.append(idx)
        return sorted(set(resolved))

    if total_layers <= 0:
        return []
    use_count = max(1, int(round(total_layers * placement_config.top_fraction)))
    start_idx = total_layers - use_count
    return list(range(start_idx, total_layers))

