from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import InferenceToolConfig
from .controller import RuleBasedMemoryController
from .memory_state import MultiScopeMemoryState
from .modeling_imm import QwenImmAdapter, QwenImmLayerWrapper


@dataclass
class SessionRecord:
    session_id: str
    memory_state: MultiScopeMemoryState
    turn_index: int = 0


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}

    def get_or_create_session(
        self,
        session_id: str,
        template_memory_state: MultiScopeMemoryState,
    ) -> SessionRecord:
        if session_id in self._sessions:
            return self._sessions[session_id]

        memory_state = MultiScopeMemoryState(
            key_dim=template_memory_state.key_dim,
            value_dim=template_memory_state.value_dim,
            working_slots=template_memory_state.working_slots,
            session_slots=template_memory_state.session_slots,
            replacement_policy=template_memory_state.replacement_policy,
        )
        record = SessionRecord(session_id=session_id, memory_state=memory_state)
        self._sessions[session_id] = record
        return record

    def save_session(self, session_id: str, file_path: str) -> None:
        record = self._sessions[session_id]
        payload = {
            "session_id": record.session_id,
            "turn_index": record.turn_index,
            "memory_state": record.memory_state.get_state_dict(),
        }
        torch.save(payload, file_path)

    def load_session(self, file_path: str) -> SessionRecord:
        payload = torch.load(file_path, map_location="cpu")
        session_id = str(payload["session_id"])
        memory_state = MultiScopeMemoryState(
            key_dim=int(payload["memory_state"]["key_dim"]),
            value_dim=int(payload["memory_state"]["value_dim"]),
            working_slots=int(payload["memory_state"]["working_slots"]),
            session_slots=int(payload["memory_state"]["session_slots"]),
        )
        memory_state.load_state_dict(payload["memory_state"])
        record = SessionRecord(
            session_id=session_id,
            memory_state=memory_state,
            turn_index=int(payload["turn_index"]),
        )
        self._sessions[session_id] = record
        return record


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        controller: RuleBasedMemoryController,
        template_memory_state: MultiScopeMemoryState,
        options: Optional[InferenceToolConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.controller = controller
        self.template_memory_state = template_memory_state
        self.options = options or InferenceToolConfig()
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
            template_memory_state=self.template_memory_state,
        )
        adapter = self._resolve_adapter(self.model)
        adapter.set_memory_state(record.memory_state)

        if self.options.reset_working_memory_per_turn:
            record.memory_state.reset_working()

        prompt_text = self._build_prompt(user_text)
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(next(self.model.parameters()).device)
        attention_mask = model_inputs["attention_mask"].to(input_ids.device)

        # In online generation we do not have supervised labels, so we keep
        # history lookup mask unset here. Training provides explicit masks.
        adapter.set_history_lookup_mask(None)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=not self.options.deterministic,
            )

        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt_text) :].strip()

        self._write_turn_summary(record, input_ids, attention_mask)
        record.turn_index += 1
        return response_text

    def _build_prompt(self, user_text: str) -> str:
        return f"User: {user_text}\nAssistant:"

    def _write_turn_summary(
        self,
        record: SessionRecord,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        adapter = self._resolve_adapter(self.model)
        wrapped_layers = adapter.wrapped_layers
        if not wrapped_layers:
            return

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            final_hidden = outputs.hidden_states[-1]
            wrapped_layers[-1].imm_module.write_session_summary(
                hidden_states=final_hidden,
                memory_state=record.memory_state,
                attention_mask=attention_mask,
                turn_index=record.turn_index,
                retention_score=1.0,
            )

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

