from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import ImmPlacementConfig, TurnSummaryConfig
from .controller import RuleBasedMemoryController


@dataclass(frozen=True)
class ImmForwardStats:
    session_attention: Optional[torch.Tensor]
    working_attention: Optional[torch.Tensor]


class TurnSummaryCompressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        summary_config: Optional[TurnSummaryConfig] = None,
    ) -> None:
        super().__init__()
        self.summary_config = summary_config or TurnSummaryConfig()
        self.summary_pooling_logits = nn.Linear(hidden_dim, 1, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim) if self.summary_config.use_layer_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.summary_config.pooling_strategy == "last_token":
            pooled = self._last_token_pool(hidden_states, attention_mask)
        elif self.summary_config.pooling_strategy == "mean_pool":
            pooled = self._mean_pool(hidden_states, attention_mask)
        elif self.summary_config.pooling_strategy == "attention_pool":
            pooled = self._attention_pool(hidden_states, attention_mask)
        else:
            raise ValueError(f"unsupported pooling strategy: {self.summary_config.pooling_strategy}")

        return self.output_norm(pooled)

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
        scores = self.summary_pooling_logits(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", weights, hidden_states)


class ImplicitMemoryModule(nn.Module):
    """
    IMM module that owns the projections for memory compress, write, read, and merge.

    New dual-stream API:
      - compress_to_kv: compress a turn's hidden states to a single K,V pair
      - query_and_merge: query accumulated K,V slots and merge into present hidden states
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
        self.write_key_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.write_value_proj = nn.Linear(hidden_dim, value_dim, bias=False)
        self.merge_norm = nn.LayerNorm(hidden_dim)
        self.summary_compressor = TurnSummaryCompressor(
            hidden_dim=hidden_dim,
            summary_config=summary_config,
        )

    def compress_to_kv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a turn's hidden states into a single K,V memory entry.

        Args:
            hidden_states: [B, T, hidden_dim]
            attention_mask: [B, T] padding mask (1=real, 0=pad)

        Returns:
            key: [B, key_dim]
            value: [B, value_dim]
        """
        summary = self.summary_compressor(hidden_states, attention_mask)
        key = self.write_key_proj(summary)
        value = self.write_value_proj(summary)
        return key, value

    def query_and_merge(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        history_lookup_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Query memory slots and merge retrieved values into hidden states.

        Args:
            hidden_states: [B, T, hidden_dim]
            memory_keys: [B, N, key_dim]
            memory_values: [B, N, value_dim]
            valid_mask: [B, N] bool, True for valid memory slots
            history_lookup_mask: [B, T] bool, True means masked (no lookup at this token)

        Returns:
            merged hidden states: [B, T, hidden_dim]
        """
        query = self.query_proj(hidden_states)  # [B, T, key_dim]
        scores = torch.einsum("btd,bnd->btn", query, memory_keys)  # [B, T, N]

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.unsqueeze(1), -1e9)

        weights = torch.softmax(scores, dim=-1)

        # Zero out attention weights when no valid slots exist for a batch item.
        if valid_mask is not None:
            any_valid = valid_mask.any(dim=-1, keepdim=True).unsqueeze(1)  # [B, 1, 1]
            weights = weights * any_valid.to(weights.dtype)

        retrieved = torch.einsum("btn,bnd->btd", weights, memory_values)  # [B, T, value_dim]

        if history_lookup_mask is not None:
            allowed = (~history_lookup_mask).unsqueeze(-1).to(retrieved.dtype)
            retrieved = retrieved * allowed

        merged = self.output_proj(retrieved)
        return self.merge_norm(hidden_states + merged)


class QwenImmLayerWrapper(nn.Module):
    """
    Wrap one decoder layer with mode-based IMM processing.

    Modes:
      - "passthrough": no IMM, base layer only
      - "history_collect": after base layer, compress hidden states → K,V
        and accumulate in per-layer buffers. Uses torch.enable_grad() so
        that the summary compressor and write projections participate in
        the computation graph even when the outer context is no_grad.
      - "present_query": after base layer, query accumulated K,V slots
        and merge retrieved memory into hidden states.
    """

    def __init__(self, base_layer: nn.Module, imm_module: ImplicitMemoryModule) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.imm_module = imm_module

        self._mode: str = "passthrough"

        # Transient per-forward buffers for collected memory slots.
        self._collected_keys: List[torch.Tensor] = []
        self._collected_values: List[torch.Tensor] = []
        self._collected_valid: List[torch.Tensor] = []

        # State for history_collect mode.
        self._history_attention_mask: Optional[torch.Tensor] = None
        self._history_row_mask: Optional[torch.Tensor] = None

        # State for present_query mode.
        self._history_lookup_mask: Optional[torch.Tensor] = None

    # ---- mode setters -------------------------------------------------------

    def set_history_collect_mode(
        self,
        attention_mask: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Switch to history collection. Must be called before each history turn forward."""
        self._mode = "history_collect"
        self._history_attention_mask = attention_mask
        self._history_row_mask = row_mask

    def set_present_query_mode(
        self,
        history_lookup_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._mode = "present_query"
        self._history_lookup_mask = history_lookup_mask

    def set_passthrough_mode(self) -> None:
        self._mode = "passthrough"

    # ---- memory slot management ----------------------------------------------

    def clear_memory_slots(self) -> None:
        """Clear all accumulated K,V from history turns."""
        self._collected_keys.clear()
        self._collected_values.clear()
        self._collected_valid.clear()

    def append_memory_slot(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Manually append a K,V entry (used by inference to add completed turns)."""
        self._collected_keys.append(key)
        self._collected_values.append(value)
        if valid_mask is not None:
            self._collected_valid.append(valid_mask)

    def get_num_memory_slots(self) -> int:
        return len(self._collected_keys)

    # ---- attribute delegation ------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as original_exc:
            base_layer = self._modules.get("base_layer")
            if base_layer is not None and hasattr(base_layer, name):
                return getattr(base_layer, name)
            raise original_exc

    # ---- forward -------------------------------------------------------------

    def forward(self, *args, **kwargs):
        base_output = self.base_layer(*args, **kwargs)

        if self._mode == "passthrough":
            return base_output

        # Extract hidden states from the base layer output.
        if isinstance(base_output, tuple):
            hidden_states = base_output[0]
        elif torch.is_tensor(base_output):
            hidden_states = base_output
        else:
            raise TypeError(
                "Unsupported decoder layer output type. Expected tensor or tuple."
            )

        if self._mode == "history_collect":
            # Re-enable gradient tracking for the compressor and write projections
            # even when the outer backbone forward runs under torch.no_grad().
            with torch.enable_grad():
                key, value = self.imm_module.compress_to_kv(
                    hidden_states.detach(),
                    self._history_attention_mask,
                )
                self._collected_keys.append(key)
                self._collected_values.append(value)
                if self._history_row_mask is not None:
                    self._collected_valid.append(self._history_row_mask)
            # History stream passes hidden states through unchanged.
            return base_output

        if self._mode == "present_query":
            if not self._collected_keys:
                return base_output

            keys = torch.stack(self._collected_keys, dim=1)    # [B, N, key_dim]
            values = torch.stack(self._collected_values, dim=1)  # [B, N, value_dim]
            valid_mask = (
                torch.stack(self._collected_valid, dim=1)
                if self._collected_valid
                else None
            )

            hidden_out = self.imm_module.query_and_merge(
                hidden_states=hidden_states,
                memory_keys=keys,
                memory_values=values,
                valid_mask=valid_mask,
                history_lookup_mask=self._history_lookup_mask,
            )

            if isinstance(base_output, tuple):
                return (hidden_out, *base_output[1:])
            return hidden_out

        return base_output


class QwenImmAdapter(nn.Module):
    """
    Adapter that injects IMM wrappers into selected decoder layers of a base model.

    Supports dual-stream training (history collect + present query) and
    single-stream inference (present query with persistent per-layer memory).
    """

    def __init__(
        self,
        base_model: nn.Module,
        placement_config: ImmPlacementConfig,
        controller: RuleBasedMemoryController,
        hidden_dim: int,
        key_dim: int,
        value_dim: int,
        summary_config: Optional[TurnSummaryConfig] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.placement_config = placement_config
        self.controller = controller
        self.summary_config = summary_config

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
            layers[layer_index] = wrapper
            self.wrapped_layers.append(wrapper)

        self.selected_layer_indices = tuple(selected_indices)

    @property
    def config(self):
        return self.base_model.config

    # ---- mode helpers (broadcast to all wrapped layers) ----------------------

    def set_history_collect_mode(
        self,
        attention_mask: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
    ) -> None:
        for wrapper in self.wrapped_layers:
            wrapper.set_history_collect_mode(attention_mask=attention_mask, row_mask=row_mask)

    def set_present_query_mode(
        self,
        history_lookup_mask: Optional[torch.Tensor] = None,
    ) -> None:
        for wrapper in self.wrapped_layers:
            wrapper.set_present_query_mode(history_lookup_mask=history_lookup_mask)

    def set_passthrough_mode(self) -> None:
        for wrapper in self.wrapped_layers:
            wrapper.set_passthrough_mode()

    def clear_all_memory_slots(self) -> None:
        for wrapper in self.wrapped_layers:
            wrapper.clear_memory_slots()

    # ---- forward -------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    # ---- generation / PEFT compatibility -------------------------------------

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if not hasattr(self.base_model, "prepare_inputs_for_generation"):
            raise AttributeError("base_model does not provide prepare_inputs_for_generation().")
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        if not hasattr(self.base_model, "_prepare_encoder_decoder_kwargs_for_generation"):
            raise AttributeError(
                "base_model does not provide _prepare_encoder_decoder_kwargs_for_generation()."
            )
        return self.base_model._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if not hasattr(self.base_model, "generate"):
            raise AttributeError("base_model does not provide generate().")
        return self.base_model.generate(*args, **kwargs)

    # ---- utilities -----------------------------------------------------------

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
