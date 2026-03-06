import torch

from imm_qwen.interfaces import MemoryReadRequest, MemoryWriteRequest
from imm_qwen.memory_state import MultiScopeMemoryState


def test_memory_write_and_read_shapes() -> None:
    state = MultiScopeMemoryState(
        key_dim=4,
        value_dim=6,
        working_slots=3,
        session_slots=5,
    )
    state.ensure_batch_size(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    key = torch.randn(2, 4)
    value = torch.randn(2, 6)
    state.write(MemoryWriteRequest(key=key, value=value, scope="session"))

    query = torch.randn(2, 7, 4)
    result = state.read(MemoryReadRequest(query=query, scope="session"))
    assert result.retrieved.shape == (2, 7, 6)

