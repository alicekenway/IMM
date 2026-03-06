import torch

from imm_qwen.controller import RuleBasedMemoryController
from imm_qwen.memory_state import MultiScopeMemoryState
from imm_qwen.modeling_imm import ImplicitMemoryModule


def test_imm_forward_shape() -> None:
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12,
        key_dim=6,
        value_dim=6,
        controller=controller,
    )
    memory_state = MultiScopeMemoryState(
        key_dim=6,
        value_dim=6,
        working_slots=4,
        session_slots=4,
    )

    hidden_states = torch.randn(2, 5, 12)
    history_lookup_mask = torch.tensor(
        [[True, True, False, False, False], [True, False, False, False, False]]
    )
    output, _ = module(
        hidden_states=hidden_states,
        memory_state=memory_state,
        history_lookup_mask=history_lookup_mask,
    )
    assert output.shape == hidden_states.shape

