import torch

from imm_qwen.controller import RuleBasedMemoryController
from imm_qwen.modeling_imm import ImplicitMemoryModule


def test_compress_to_kv_shape() -> None:
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12,
        key_dim=6,
        value_dim=6,
        controller=controller,
    )

    hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.ones(2, 5)
    key, value = module.compress_to_kv(hidden_states, attention_mask)
    assert key.shape == (2, 6)
    assert value.shape == (2, 6)


def test_query_and_merge_shape() -> None:
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12,
        key_dim=6,
        value_dim=6,
        controller=controller,
    )

    hidden_states = torch.randn(2, 5, 12)
    memory_keys = torch.randn(2, 3, 6)
    memory_values = torch.randn(2, 3, 6)
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])
    history_lookup_mask = torch.tensor(
        [[True, True, False, False, False], [True, False, False, False, False]]
    )

    output = module.query_and_merge(
        hidden_states=hidden_states,
        memory_keys=memory_keys,
        memory_values=memory_values,
        valid_mask=valid_mask,
        history_lookup_mask=history_lookup_mask,
    )
    assert output.shape == hidden_states.shape


def test_query_and_merge_no_valid_slots() -> None:
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12,
        key_dim=6,
        value_dim=6,
        controller=controller,
    )

    hidden_states = torch.randn(2, 5, 12)
    memory_keys = torch.randn(2, 3, 6)
    memory_values = torch.randn(2, 3, 6)
    valid_mask = torch.zeros(2, 3, dtype=torch.bool)

    output = module.query_and_merge(
        hidden_states=hidden_states,
        memory_keys=memory_keys,
        memory_values=memory_values,
        valid_mask=valid_mask,
    )
    assert output.shape == hidden_states.shape
