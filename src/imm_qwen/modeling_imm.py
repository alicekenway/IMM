from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import ImmPlacementConfig, TurnSummaryConfig
from .controller import RuleBasedMemoryController


def _call_mask_factory(factory, **kwargs):
    """Adapt to transformers masking helpers whose kwargs vary by version."""
    supported = inspect.signature(factory).parameters
    if "input_embeds" in supported and "input_embeds" not in kwargs and "inputs_embeds" in kwargs:
        kwargs["input_embeds"] = kwargs["inputs_embeds"]
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return factory(**filtered_kwargs)


# ---------------------------------------------------------------------------
# TurnSummaryCompressor
# ---------------------------------------------------------------------------

class TurnSummaryCompressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        summary_config: Optional[TurnSummaryConfig] = None,
    ) -> None:
        super().__init__()
        self.summary_config = summary_config or TurnSummaryConfig()
        self.summary_pooling_logits = nn.Linear(hidden_dim, 1, bias=False)
        self.output_norm = (
            nn.LayerNorm(hidden_dim)
            if self.summary_config.use_layer_norm
            else nn.Identity()
        )

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
            raise ValueError(
                f"unsupported pooling strategy: {self.summary_config.pooling_strategy}"
            )
        return self.output_norm(pooled)

    @staticmethod
    def _last_token_pool(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        lengths = attention_mask.to(torch.long).sum(dim=1).clamp(min=1)
        last_indices = lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _attention_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = self.summary_pooling_logits(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", weights, hidden_states)


# ---------------------------------------------------------------------------
# ImplicitMemoryModule
# ---------------------------------------------------------------------------

class ImplicitMemoryModule(nn.Module):
    """Per-layer IMM: compress history turns to K,V, query from present."""

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
        """Compress hidden states into a single K,V memory entry per sample.

        Args:
            hidden_states: [B, T, hidden_dim]
            attention_mask: [B, T]  (1=real, 0=pad)

        Returns:
            key:   [B, key_dim]
            value: [B, value_dim]
        """
        summary = self.summary_compressor(hidden_states, attention_mask)
        return self.write_key_proj(summary), self.write_value_proj(summary)

    def query_and_merge(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        history_lookup_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Query memory slots and merge into present hidden states.

        Args:
            hidden_states:       [B, T, hidden_dim]
            memory_keys:         [B, N, key_dim]
            memory_values:       [B, N, value_dim]
            valid_mask:          [B, N] bool — True = valid slot
            history_lookup_mask: [B, T] bool — True = masked (no lookup)

        Returns:
            [B, T, hidden_dim]
        """
        query = self.query_proj(hidden_states)                       # [B, T, key_dim]
        scores = torch.einsum("btd,bnd->btn", query, memory_keys)   # [B, T, N]

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.unsqueeze(1), -1e9)

        weights = torch.softmax(scores, dim=-1)

        if valid_mask is not None:
            any_valid = valid_mask.any(dim=-1, keepdim=True).unsqueeze(1)  # [B,1,1]
            weights = weights * any_valid.to(weights.dtype)

        retrieved = torch.einsum("btn,bnd->btd", weights, memory_values)  # [B, T, value_dim]

        if history_lookup_mask is not None:
            allowed = (~history_lookup_mask).unsqueeze(-1).to(retrieved.dtype)
            retrieved = retrieved * allowed

        merged = self.output_proj(retrieved)
        return self.merge_norm(hidden_states + merged)


# ---------------------------------------------------------------------------
# QwenImmLayerWrapper  (used by inference mode-based path)
# ---------------------------------------------------------------------------

class QwenImmLayerWrapper(nn.Module):
    """Wraps one decoder layer.  Mode-based operation for inference;
    the training path (dual_stream_forward) calls base_layer and
    imm_module directly and does not use the mode system."""

    def __init__(
        self,
        base_layer: nn.Module,
        imm_module: ImplicitMemoryModule,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.imm_module = imm_module

        # Inference mode state
        self._mode: str = "passthrough"
        self._collected_keys: List[torch.Tensor] = []
        self._collected_values: List[torch.Tensor] = []
        self._collected_valid: List[torch.Tensor] = []
        self._history_attention_mask: Optional[torch.Tensor] = None
        self._history_row_mask: Optional[torch.Tensor] = None
        self._history_lookup_mask: Optional[torch.Tensor] = None

    # ---- mode setters (inference only) --------------------------------------

    def set_history_collect_mode(
        self,
        attention_mask: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
    ) -> None:
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

    def clear_memory_slots(self) -> None:
        self._collected_keys.clear()
        self._collected_values.clear()
        self._collected_valid.clear()

    def append_memory_slot(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._collected_keys.append(key)
        self._collected_values.append(value)
        if valid_mask is not None:
            self._collected_valid.append(valid_mask)

    def get_num_memory_slots(self) -> int:
        return len(self._collected_keys)

    # ---- attribute delegation -----------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as original_exc:
            base_layer = self._modules.get("base_layer")
            if base_layer is not None and hasattr(base_layer, name):
                return getattr(base_layer, name)
            raise original_exc

    # ---- forward (inference modes) ------------------------------------------

    def forward(self, *args, **kwargs):
        base_output = self.base_layer(*args, **kwargs)

        if self._mode == "passthrough":
            return base_output

        if isinstance(base_output, tuple):
            hidden_states = base_output[0]
        elif torch.is_tensor(base_output):
            hidden_states = base_output
        else:
            raise TypeError("Unsupported decoder layer output type.")

        if self._mode == "history_collect":
            with torch.enable_grad():
                key, value = self.imm_module.compress_to_kv(
                    hidden_states.detach(),
                    self._history_attention_mask,
                )
                self._collected_keys.append(key)
                self._collected_values.append(value)
                if self._history_row_mask is not None:
                    self._collected_valid.append(self._history_row_mask)
            return base_output

        if self._mode == "present_query":
            if not self._collected_keys:
                return base_output
            keys = torch.stack(self._collected_keys, dim=1)
            values = torch.stack(self._collected_values, dim=1)
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


# ---------------------------------------------------------------------------
# QwenImmAdapter
# ---------------------------------------------------------------------------

class QwenImmAdapter(nn.Module):
    """Injects IMM wrappers into selected decoder layers of a Qwen model.

    Training uses ``dual_stream_forward`` which manually iterates through
    layers so that all IMM parameters appear in a single DDP-tracked
    forward pass.  Inference uses the mode-based wrapper mechanism via
    the normal ``forward`` → ``base_model`` delegation.
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

    # ---- forward dispatch ---------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # dual-stream training args (passed via **kwargs through PEFT / DDP)
        history_input_ids: Optional[torch.Tensor] = None,
        history_attention_mask: Optional[torch.Tensor] = None,
        history_line_mask: Optional[torch.Tensor] = None,
        history_lookup_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if history_input_ids is not None:
            return self.dual_stream_forward(
                history_input_ids=history_input_ids,
                history_attention_mask=history_attention_mask,
                history_line_mask=history_line_mask,
                present_input_ids=input_ids,
                present_attention_mask=attention_mask,
                history_lookup_mask=history_lookup_mask,
            )
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    # ---- dual-stream training forward ---------------------------------------

    def dual_stream_forward(
        self,
        history_input_ids: torch.Tensor,       # [B, H, T_h]
        history_attention_mask: torch.Tensor,   # [B, H, T_h]
        history_line_mask: torch.Tensor,        # [B, H]
        present_input_ids: torch.Tensor,        # [B, T_p]
        present_attention_mask: torch.Tensor,   # [B, T_p]
        history_lookup_mask: Optional[torch.Tensor] = None,  # [B, T_p]
    ):
        """Layer-by-layer dual-stream forward.

        History turns are batched as ``B*H`` and processed through each
        layer in one tensor op (no Python loop over turns).  At every IMM
        layer the history hidden states are compressed to K,V and the
        present stream queries them.  K,V are temporary and discarded
        after each layer.

        All IMM parameters (compressor, write projections, query
        projection, output projection, merge norm) are called inside this
        single forward pass, making it safe for DDP with
        ``find_unused_parameters=False``.
        """
        from transformers.masking_utils import create_causal_mask
        from transformers.modeling_outputs import CausalLMOutputWithPast

        B, H, T_h = history_input_ids.shape
        device = present_input_ids.device

        # --- fast path: no history -------------------------------------------
        if H == 0 or not history_line_mask.any():
            return self.base_model(
                input_ids=present_input_ids,
                attention_mask=present_attention_mask,
                use_cache=False,
                return_dict=True,
            )

        B_H = B * H
        T_p = present_input_ids.size(1)

        # --- extract Qwen internals -----------------------------------------
        qwen: nn.Module = self.base_model.model        # Qwen2Model
        embed_tokens = qwen.embed_tokens
        all_layers: nn.ModuleList = qwen.layers         # some are wrappers
        final_norm = qwen.norm
        rotary_emb = qwen.rotary_emb
        lm_head = self.base_model.lm_head
        config = self.base_model.config

        # --- embed -----------------------------------------------------------
        hist_ids_flat = history_input_ids.reshape(B_H, T_h)
        hist_mask_flat = history_attention_mask.reshape(B_H, T_h)

        with torch.no_grad():
            hist_hidden = embed_tokens(hist_ids_flat)    # [B*H, T_h, D]
        pres_hidden = embed_tokens(present_input_ids)    # [B, T_p, D]

        # --- positions -------------------------------------------------------
        hist_cache_pos = torch.arange(T_h, device=device)
        hist_pos_ids = hist_cache_pos.unsqueeze(0).expand(B_H, -1)
        pres_cache_pos = torch.arange(T_p, device=device)
        pres_pos_ids = pres_cache_pos.unsqueeze(0).expand(B, -1)

        # --- rotary embeddings (computed once, reused across layers) ---------
        hist_pos_emb = rotary_emb(hist_hidden, hist_pos_ids)
        pres_pos_emb = rotary_emb(pres_hidden, pres_pos_ids)

        # --- causal masks (computed once) ------------------------------------
        _mask_kw_hist = dict(
            config=config, inputs_embeds=hist_hidden,
            attention_mask=hist_mask_flat, cache_position=hist_cache_pos,
            past_key_values=None, position_ids=hist_pos_ids,
        )
        _mask_kw_pres = dict(
            config=config, inputs_embeds=pres_hidden,
            attention_mask=present_attention_mask, cache_position=pres_cache_pos,
            past_key_values=None, position_ids=pres_pos_ids,
        )
        hist_masks = {
            "full_attention": _call_mask_factory(create_causal_mask, **_mask_kw_hist)
        }
        pres_masks = {
            "full_attention": _call_mask_factory(create_causal_mask, **_mask_kw_pres)
        }

        if getattr(qwen, "has_sliding_layers", False):
            from transformers.masking_utils import create_sliding_window_causal_mask
            hist_masks["sliding_attention"] = _call_mask_factory(
                create_sliding_window_causal_mask, **_mask_kw_hist
            )
            pres_masks["sliding_attention"] = _call_mask_factory(
                create_sliding_window_causal_mask, **_mask_kw_pres
            )

        # --- shared layer kwargs (no cache) ----------------------------------
        def _layer_kw(mask_dict, mask_key, pos_ids, pos_emb, cache_pos):
            return dict(
                attention_mask=mask_dict.get(mask_key, mask_dict["full_attention"]),
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_pos,
            )

        # --- layer-by-layer --------------------------------------------------
        for decoder_layer in all_layers:
            is_imm = isinstance(decoder_layer, QwenImmLayerWrapper)
            base = decoder_layer.base_layer if is_imm else decoder_layer
            attn_type = getattr(base, "attention_type", "full_attention")

            hist_kw = _layer_kw(hist_masks, attn_type, hist_pos_ids, hist_pos_emb, hist_cache_pos)
            pres_kw = _layer_kw(pres_masks, attn_type, pres_pos_ids, pres_pos_emb, pres_cache_pos)

            # history stream — backbone under no_grad
            with torch.no_grad():
                hist_hidden = base(hist_hidden, **hist_kw)

            if is_imm:
                # compress all B*H turns in one batched call (no loop over H)
                keys, values = decoder_layer.imm_module.compress_to_kv(
                    hist_hidden.detach(), hist_mask_flat,
                )
                # reshape to [B, H, dim] so query_and_merge sees N=H slots
                keys = keys.view(B, H, -1)
                values = values.view(B, H, -1)

                # present stream — base layer with grad
                pres_hidden = base(pres_hidden, **pres_kw)

                # query history K,V and merge into present
                pres_hidden = decoder_layer.imm_module.query_and_merge(
                    hidden_states=pres_hidden,
                    memory_keys=keys,
                    memory_values=values,
                    valid_mask=history_line_mask,
                    history_lookup_mask=history_lookup_mask,
                )
            else:
                # non-IMM layer: just run present through
                pres_hidden = decoder_layer(pres_hidden, **pres_kw)

        # --- final head ------------------------------------------------------
        pres_hidden = final_norm(pres_hidden)
        logits = lm_head(pres_hidden)

        return CausalLMOutputWithPast(logits=logits)

    # ---- inference mode helpers (broadcast) ---------------------------------

    def set_history_collect_mode(
        self,
        attention_mask: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
    ) -> None:
        for w in self.wrapped_layers:
            w.set_history_collect_mode(attention_mask=attention_mask, row_mask=row_mask)

    def set_present_query_mode(
        self,
        history_lookup_mask: Optional[torch.Tensor] = None,
    ) -> None:
        for w in self.wrapped_layers:
            w.set_present_query_mode(history_lookup_mask=history_lookup_mask)

    def set_passthrough_mode(self) -> None:
        for w in self.wrapped_layers:
            w.set_passthrough_mode()

    def clear_all_memory_slots(self) -> None:
        for w in self.wrapped_layers:
            w.clear_memory_slots()

    # ---- generation / PEFT compatibility ------------------------------------

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

    # ---- utilities ----------------------------------------------------------

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
            maybe = getattr(model, "layers")
            if isinstance(maybe, nn.ModuleList):
                candidates.append(("layers", maybe))
        if not candidates:
            raise ValueError(
                "Could not locate decoder layers. "
                "Expected one of: model.layers, transformer.h, layers."
            )
        return candidates[0][1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
