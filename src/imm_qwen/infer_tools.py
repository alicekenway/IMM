from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import InferenceToolConfig
from .controller import RuleBasedMemoryController
from .modeling_imm import QwenImmAdapter, QwenImmLayerWrapper


@dataclass
class LayerSlotBank:
    """Persistent per-layer K,V memory bank for inference sessions."""
    keys: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)

    @property
    def num_slots(self) -> int:
        return len(self.keys)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.keys.append(key.detach().cpu())
        self.values.append(value.detach().cpu())

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()


@dataclass
class SessionRecord:
    session_id: str
    layer_banks: List[LayerSlotBank] = field(default_factory=list)
    turn_index: int = 0


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}

    def get_or_create_session(
        self,
        session_id: str,
        num_layers: int,
    ) -> SessionRecord:
        if session_id in self._sessions:
            return self._sessions[session_id]

        record = SessionRecord(
            session_id=session_id,
            layer_banks=[LayerSlotBank() for _ in range(num_layers)],
        )
        self._sessions[session_id] = record
        return record

    def save_session(self, session_id: str, file_path: str) -> None:
        record = self._sessions[session_id]
        payload = {
            "session_id": record.session_id,
            "turn_index": record.turn_index,
            "num_layers": len(record.layer_banks),
            "layer_banks": [
                {
                    "keys": [k.clone() for k in bank.keys],
                    "values": [v.clone() for v in bank.values],
                }
                for bank in record.layer_banks
            ],
        }
        torch.save(payload, file_path)

    def load_session(self, file_path: str) -> SessionRecord:
        payload = torch.load(file_path, map_location="cpu")
        session_id = str(payload["session_id"])
        num_layers = int(payload["num_layers"])
        layer_banks = []
        for bank_data in payload["layer_banks"]:
            bank = LayerSlotBank(
                keys=bank_data["keys"],
                values=bank_data["values"],
            )
            layer_banks.append(bank)
        record = SessionRecord(
            session_id=session_id,
            layer_banks=layer_banks,
            turn_index=int(payload["turn_index"]),
        )
        self._sessions[session_id] = record
        return record


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        controller: RuleBasedMemoryController,
        options: Optional[InferenceToolConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.controller = controller
        self.options = options or InferenceToolConfig()

        adapter = self._resolve_adapter(self.model)
        self.num_imm_layers = len(adapter.wrapped_layers)
        self.session_manager = SessionManager()

    def generate_response(
        self,
        session_id: str,
        user_text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
    ) -> str:
        record = self.session_manager.get_or_create_session(
            session_id=session_id,
            num_layers=self.num_imm_layers,
        )
        adapter = self._resolve_adapter(self.model)
        device = next(self.model.parameters()).device

        # Load persistent session memory into the wrapper layer buffers.
        self._load_session_memory_to_wrappers(adapter, record, device)

        prompt_text = self._build_prompt(user_text)
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        # Query existing memory during generation.
        adapter.set_present_query_mode(history_lookup_mask=None)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=not self.options.deterministic,
            )

        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt_text):].strip()

        # Write the completed turn into per-layer memory.
        self._write_turn_to_session(adapter, record, input_ids, attention_mask, device)
        record.turn_index += 1

        adapter.set_passthrough_mode()
        return response_text

    def _build_prompt(self, user_text: str) -> str:
        return f"User: {user_text}\nAssistant:"

    def _load_session_memory_to_wrappers(
        self,
        adapter: QwenImmAdapter,
        record: SessionRecord,
        device: torch.device,
    ) -> None:
        """Populate each wrapper's K,V buffers from the persistent session banks."""
        for wrapper, bank in zip(adapter.wrapped_layers, record.layer_banks):
            wrapper.clear_memory_slots()
            for key, value in zip(bank.keys, bank.values):
                wrapper.append_memory_slot(
                    key=key.to(device),
                    value=value.to(device),
                )

    def _write_turn_to_session(
        self,
        adapter: QwenImmAdapter,
        record: SessionRecord,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Run the completed turn through the backbone in history_collect mode
        to produce K,V for each IMM layer, then persist them."""
        # Temporarily clear and use collect mode to capture this turn's K,V.
        # Save existing slots first.
        saved_keys = [list(w._collected_keys) for w in adapter.wrapped_layers]
        saved_values = [list(w._collected_values) for w in adapter.wrapped_layers]
        saved_valid = [list(w._collected_valid) for w in adapter.wrapped_layers]

        for wrapper in adapter.wrapped_layers:
            wrapper.clear_memory_slots()

        adapter.set_history_collect_mode(attention_mask=attention_mask)

        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        # Extract the newly collected K,V and persist to session banks.
        for wrapper, bank in zip(adapter.wrapped_layers, record.layer_banks):
            if wrapper._collected_keys:
                bank.append(
                    key=wrapper._collected_keys[0],
                    value=wrapper._collected_values[0],
                )

        # Restore the previous slots.
        for wrapper, keys, values, valid in zip(
            adapter.wrapped_layers, saved_keys, saved_values, saved_valid
        ):
            wrapper._collected_keys = keys
            wrapper._collected_values = values
            wrapper._collected_valid = valid

    @staticmethod
    def _resolve_adapter(model: torch.nn.Module) -> QwenImmAdapter:
        if isinstance(model, QwenImmAdapter):
            return model
        if hasattr(model, "base_model"):
            base_model = getattr(model, "base_model")
            if isinstance(base_model, QwenImmAdapter):
                return base_model
            if hasattr(base_model, "model") and isinstance(base_model.model, QwenImmAdapter):
                return base_model.model
        raise ValueError("Cannot resolve QwenImmAdapter from model wrapper stack.")
