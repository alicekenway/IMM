from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .interfaces import (
    MemoryReadRequest,
    MemoryReadResult,
    MemoryScope,
    MemoryWriteRequest,
    ReplacementPolicyProtocol,
)


@dataclass
class MemoryBankTensors:
    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor
    write_pointer: torch.Tensor
    turn_index: torch.Tensor
    retention_score: torch.Tensor


class FifoReplacementPolicy(ReplacementPolicyProtocol):
    def select_slot_indices(
        self,
        write_pointer: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del valid_mask
        return write_pointer


class MultiScopeMemoryState:
    """
    Runtime memory state with two scopes:
      - working memory
      - session memory

    Design goals:
      - pre-allocated bank tensors
      - in-place slot updates
      - no required host-device transfer during normal read/write
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        working_slots: int,
        session_slots: int,
        replacement_policy: Optional[ReplacementPolicyProtocol] = None,
    ) -> None:
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.working_slots = working_slots
        self.session_slots = session_slots
        self.replacement_policy = replacement_policy or FifoReplacementPolicy()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        self.batch_size = 0
        self.working_bank: Optional[MemoryBankTensors] = None
        self.session_bank: Optional[MemoryBankTensors] = None

    def ensure_batch_size(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            self.batch_size == batch_size
            and self.device == device
            and self.dtype == dtype
            and self.working_bank is not None
            and self.session_bank is not None
        ):
            return

        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.working_bank = self._create_bank(
            batch_size=batch_size,
            num_slots=self.working_slots,
            device=device,
            dtype=dtype,
        )
        self.session_bank = self._create_bank(
            batch_size=batch_size,
            num_slots=self.session_slots,
            device=device,
            dtype=dtype,
        )

    def reset_working(self) -> None:
        if self.working_bank is None:
            return
        self._reset_bank(self.working_bank)

    def reset_session(self) -> None:
        if self.session_bank is None:
            return
        self._reset_bank(self.session_bank)

    def read(self, request: MemoryReadRequest) -> MemoryReadResult:
        bank = self._get_bank(request.scope)
        query = request.query
        scores = torch.einsum("btd,bnd->btn", query, bank.keys)
        weights = self._masked_softmax(scores, bank.valid_mask)
        retrieved = torch.einsum("btn,bnd->btd", weights, bank.values)

        if request.history_lookup_mask is not None:
            allowed_mask = (~request.history_lookup_mask).unsqueeze(-1).to(retrieved.dtype)
            retrieved = retrieved * allowed_mask

        return MemoryReadResult(retrieved=retrieved, attention_weights=weights)

    def write(self, request: MemoryWriteRequest) -> None:
        bank = self._get_bank(request.scope)
        keys = request.key
        values = request.value
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("write key/value must have shape [batch_size, dim].")
        if keys.shape[0] != self.batch_size or values.shape[0] != self.batch_size:
            raise ValueError("write key/value batch size mismatch with memory state.")

        slot_indices = self.replacement_policy.select_slot_indices(
            write_pointer=bank.write_pointer,
            valid_mask=bank.valid_mask,
        )
        batch_indices = torch.arange(self.batch_size, device=self.device)
        if request.row_mask is None:
            row_mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        else:
            row_mask = request.row_mask.to(device=self.device, dtype=torch.bool)
            if row_mask.ndim != 1 or row_mask.shape[0] != self.batch_size:
                raise ValueError("row_mask must have shape [batch_size].")

        if not torch.any(row_mask):
            return
        active_batch = batch_indices[row_mask]
        active_slots = slot_indices[row_mask]

        bank.keys[active_batch, active_slots] = keys[row_mask]
        bank.values[active_batch, active_slots] = values[row_mask]
        bank.valid_mask[active_batch, active_slots] = True
        bank.write_pointer[active_batch] = (active_slots + 1) % bank.keys.size(1)

        if request.metadata is not None:
            if request.metadata.turn_index is not None:
                bank.turn_index[active_batch, active_slots] = int(request.metadata.turn_index)
            if request.metadata.retention_score is not None:
                bank.retention_score[active_batch, active_slots] = float(
                    request.metadata.retention_score
                )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "working_slots": self.working_slots,
            "session_slots": self.session_slots,
            "working_bank": self._bank_to_state(self.working_bank),
            "session_bank": self._bank_to_state(self.session_bank),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.batch_size = int(state_dict["batch_size"])
        self.key_dim = int(state_dict["key_dim"])
        self.value_dim = int(state_dict["value_dim"])
        self.working_slots = int(state_dict["working_slots"])
        self.session_slots = int(state_dict["session_slots"])

        working_state = state_dict["working_bank"]
        session_state = state_dict["session_bank"]
        if working_state is not None:
            self.working_bank = self._state_to_bank(working_state)
        if session_state is not None:
            self.session_bank = self._state_to_bank(session_state)
        if self.working_bank is not None:
            self.device = self.working_bank.keys.device
            self.dtype = self.working_bank.keys.dtype

    def _get_bank(self, scope: MemoryScope) -> MemoryBankTensors:
        if scope == "working":
            if self.working_bank is None:
                raise RuntimeError("working memory bank is not initialized.")
            return self.working_bank
        if scope == "session":
            if self.session_bank is None:
                raise RuntimeError("session memory bank is not initialized.")
            return self.session_bank
        raise ValueError(f"unknown memory scope: {scope}")

    def _create_bank(
        self,
        batch_size: int,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MemoryBankTensors:
        keys = torch.zeros(batch_size, num_slots, self.key_dim, device=device, dtype=dtype)
        values = torch.zeros(batch_size, num_slots, self.value_dim, device=device, dtype=dtype)
        valid_mask = torch.zeros(batch_size, num_slots, device=device, dtype=torch.bool)
        write_pointer = torch.zeros(batch_size, device=device, dtype=torch.long)
        turn_index = torch.full(
            (batch_size, num_slots),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )
        retention_score = torch.zeros(batch_size, num_slots, device=device, dtype=dtype)
        return MemoryBankTensors(
            keys=keys,
            values=values,
            valid_mask=valid_mask,
            write_pointer=write_pointer,
            turn_index=turn_index,
            retention_score=retention_score,
        )

    @staticmethod
    def _reset_bank(bank: MemoryBankTensors) -> None:
        bank.keys.zero_()
        bank.values.zero_()
        bank.valid_mask.zero_()
        bank.write_pointer.zero_()
        bank.turn_index.fill_(-1)
        bank.retention_score.zero_()

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # scores: [B, T, N], valid_mask: [B, N]
        expanded_mask = valid_mask.unsqueeze(1)  # [B, 1, N]
        masked_scores = scores.masked_fill(~expanded_mask, -1e9)
        weights = F.softmax(masked_scores, dim=-1)

        # If a batch item has no valid slots, set weights to zero instead of NaN-like behavior.
        valid_any = valid_mask.any(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        weights = weights * valid_any.to(weights.dtype)
        return weights

    @staticmethod
    def _bank_to_state(bank: Optional[MemoryBankTensors]) -> Optional[Dict[str, torch.Tensor]]:
        if bank is None:
            return None
        return {
            "keys": bank.keys.detach().cpu(),
            "values": bank.values.detach().cpu(),
            "valid_mask": bank.valid_mask.detach().cpu(),
            "write_pointer": bank.write_pointer.detach().cpu(),
            "turn_index": bank.turn_index.detach().cpu(),
            "retention_score": bank.retention_score.detach().cpu(),
        }

    @staticmethod
    def _state_to_bank(state: Dict[str, torch.Tensor]) -> MemoryBankTensors:
        return MemoryBankTensors(
            keys=state["keys"],
            values=state["values"],
            valid_mask=state["valid_mask"],
            write_pointer=state["write_pointer"],
            turn_index=state["turn_index"],
            retention_score=state["retention_score"],
        )

