import torch

from imm_qwen.controller import RuleBasedMemoryController


def test_build_history_lookup_mask_from_labels() -> None:
    controller = RuleBasedMemoryController()
    labels = torch.tensor([[-100, -100, 15, 16, 17]])
    mask = controller.build_history_lookup_mask(labels)
    expected = torch.tensor([[True, True, False, False, False]])
    assert torch.equal(mask, expected)

