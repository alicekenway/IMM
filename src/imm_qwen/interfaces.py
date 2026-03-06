from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Tuple

import torch


MemoryScope = Literal["working", "session"]


@dataclass(frozen=True)
class MemoryMetadata:
    turn_index: Optional[int] = None
    speaker: Optional[str] = None
    timestamp: Optional[float] = None
    retention_score: Optional[float] = None


@dataclass(frozen=True)
class MemoryReadRequest:
    query: torch.Tensor
    scope: MemoryScope
    # True means "masked out / do not read history here".
    history_lookup_mask: Optional[torch.Tensor] = None
    # Optional additional slot mask can be used by higher-level logic.
    slot_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MemoryReadResult:
    retrieved: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MemoryWriteRequest:
    key: torch.Tensor
    value: torch.Tensor
    scope: MemoryScope
    metadata: Optional[MemoryMetadata] = None
    row_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class StepContext:
    is_prefill: bool
    is_turn_end: bool
    token_index: int
    has_control_span: bool
    in_control_span: bool


class ReplacementPolicyProtocol(Protocol):
    def select_slot_indices(
        self,
        write_pointer: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return slot indices with shape [batch_size]."""


class MemoryStateProtocol(Protocol):
    device: torch.device

    def reset_working(self) -> None:
        ...

    def reset_session(self) -> None:
        ...

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        ...

    def write(self, request: MemoryWriteRequest) -> None:
        ...

    def get_state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


class MemoryControllerProtocol(Protocol):
    def build_history_lookup_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Return mask with shape [batch_size, sequence_length]."""

    def should_read_memory(
        self,
        scope: MemoryScope,
        step_context: StepContext,
    ) -> bool:
        ...

    def should_write_memory(self, step_context: StepContext) -> bool:
        ...

    def merge_gate(
        self,
        hidden_states: torch.Tensor,
        retrieved_states: torch.Tensor,
        scope: MemoryScope,
        history_lookup_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return gated retrieved states with same shape as retrieved_states."""


class QwenImmWrapperProtocol(Protocol):
    def set_memory_state(self, memory_state: MemoryStateProtocol) -> None:
        ...

    def set_history_lookup_mask(self, history_lookup_mask: Optional[torch.Tensor]) -> None:
        ...

