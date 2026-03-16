import torch

from imm_qwen.controller import RuleBasedMemoryController
from imm_qwen.modeling_imm import ImplicitMemoryModule


def test_compress_to_kv_shape() -> None:
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12, key_dim=6, value_dim=6, controller=controller,
    )
    hidden = torch.randn(2, 5, 12)
    mask = torch.ones(2, 5)
    key, value = module.compress_to_kv(hidden, mask)
    assert key.shape == (2, 6)
    assert value.shape == (2, 6)


def test_compress_to_kv_batched_bh() -> None:
    """Verify B*H batching produces correct shapes."""
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12, key_dim=6, value_dim=6, controller=controller,
    )
    B, H, T = 2, 4, 5
    hidden = torch.randn(B * H, T, 12)
    mask = torch.ones(B * H, T)
    key, value = module.compress_to_kv(hidden, mask)
    assert key.shape == (B * H, 6)
    key = key.view(B, H, 6)
    value = value.view(B, H, 6)
    assert key.shape == (B, H, 6)


def test_query_and_merge_shape() -> None:
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12, key_dim=6, value_dim=6, controller=controller,
    )
    hidden = torch.randn(2, 5, 12)
    mem_keys = torch.randn(2, 3, 6)
    mem_values = torch.randn(2, 3, 6)
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])
    lookup_mask = torch.tensor(
        [[True, True, False, False, False],
         [True, False, False, False, False]]
    )
    out = module.query_and_merge(
        hidden_states=hidden,
        memory_keys=mem_keys,
        memory_values=mem_values,
        valid_mask=valid_mask,
        history_lookup_mask=lookup_mask,
    )
    assert out.shape == hidden.shape


def test_query_and_merge_no_valid_slots() -> None:
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12, key_dim=6, value_dim=6, controller=controller,
    )
    hidden = torch.randn(2, 5, 12)
    mem_keys = torch.randn(2, 3, 6)
    mem_values = torch.randn(2, 3, 6)
    valid_mask = torch.zeros(2, 3, dtype=torch.bool)
    out = module.query_and_merge(
        hidden_states=hidden,
        memory_keys=mem_keys,
        memory_values=mem_values,
        valid_mask=valid_mask,
    )
    assert out.shape == hidden.shape


def test_write_projections_get_gradients() -> None:
    """Core invariant: write projections receive gradients through
    the compress → query → loss path."""
    controller = RuleBasedMemoryController()
    module = ImplicitMemoryModule(
        hidden_dim=12, key_dim=6, value_dim=6, controller=controller,
    )
    # Simulate: history hidden states (detached, as from no_grad backbone)
    hist_hidden = torch.randn(2, 5, 12)  # no requires_grad
    hist_mask = torch.ones(2, 5)
    keys, values = module.compress_to_kv(hist_hidden, hist_mask)

    # Simulate: present hidden states (with grad)
    pres_hidden = torch.randn(2, 7, 12, requires_grad=True)
    out = module.query_and_merge(
        hidden_states=pres_hidden,
        memory_keys=keys.unsqueeze(0).expand(2, -1, -1) if keys.dim() == 1 else keys.unsqueeze(1),
        memory_values=values.unsqueeze(0).expand(2, -1, -1) if values.dim() == 1 else values.unsqueeze(1),
    )

    loss = out.sum()
    loss.backward()

    assert module.write_key_proj.weight.grad is not None
    assert module.write_value_proj.weight.grad is not None
    assert module.query_proj.weight.grad is not None
    assert module.output_proj.weight.grad is not None
    assert module.write_key_proj.weight.grad.abs().sum() > 0
    assert module.write_value_proj.weight.grad.abs().sum() > 0
