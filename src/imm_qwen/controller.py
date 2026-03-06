from dataclasses import dataclass
from typing import Optional

import torch

from .config import MemoryControllerConfig
from .interfaces import MemoryScope, StepContext


@dataclass(frozen=True)
class ControllerRuntimeFlags:
    force_disable_session_read: bool = False
    force_disable_working_read: bool = False


class RuleBasedMemoryController:
    """
    First-pass memory controller with deterministic behavior.

    The design intentionally starts with explicit rules so behavior is easier
    to validate before introducing learned gating.
    """

    def __init__(
        self,
        config: Optional[MemoryControllerConfig] = None,
        runtime_flags: Optional[ControllerRuntimeFlags] = None,
    ) -> None:
        self.config = config or MemoryControllerConfig()
        self.runtime_flags = runtime_flags or ControllerRuntimeFlags()

    def build_history_lookup_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Build lookup mask from supervised labels.

        Convention:
          - True  => masked (do NOT read history memory)
          - False => unmasked (ALLOWED to read history memory)

        With standard SFT labels, prompt tokens are -100 and target tokens are
        actual ids, so this means:
          - prompt side does not query history
          - output side can query history
        """
        return labels.eq(-100)

    def should_read_memory(
        self,
        scope: MemoryScope,
        step_context: StepContext,
    ) -> bool:
        if scope == "session":
            if self.runtime_flags.force_disable_session_read:
                return False
            return True

        if scope == "working":
            if self.runtime_flags.force_disable_working_read:
                return False
            return bool(self.config.use_working_memory)

        raise ValueError(f"unknown memory scope: {scope}")

    def should_write_memory(self, step_context: StepContext) -> bool:
        return bool(step_context.is_turn_end)

    def merge_gate(
        self,
        hidden_states: torch.Tensor,
        retrieved_states: torch.Tensor,
        scope: MemoryScope,
        history_lookup_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del hidden_states
        if scope == "session":
            gate = self.config.session_merge_gate
            if history_lookup_mask is not None:
                allowed_mask = (~history_lookup_mask).unsqueeze(-1).to(retrieved_states.dtype)
                return retrieved_states * gate * allowed_mask
            return retrieved_states * gate

        if scope == "working":
            if not self.config.use_working_memory:
                return torch.zeros_like(retrieved_states)
            gate = self.config.working_merge_gate
            if history_lookup_mask is not None:
                allowed_mask = (~history_lookup_mask).unsqueeze(-1).to(retrieved_states.dtype)
                return retrieved_states * gate * allowed_mask
            return retrieved_states * gate

        raise ValueError(f"unknown memory scope: {scope}")

