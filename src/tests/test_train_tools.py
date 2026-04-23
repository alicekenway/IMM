import pytest
import torch

from imm_qwen.config import TrainingToolConfig
from imm_qwen.train_tools import (
    build_constant_lr_with_warmup_scheduler,
    build_optimizer_groups,
    load_scheduler_state,
)


class DummyTrainableModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("lora_adapter_weight", torch.nn.Parameter(torch.ones(1)))
        self.register_parameter("imm_module_gate", torch.nn.Parameter(torch.ones(1)))
        self.register_parameter("output_head", torch.nn.Parameter(torch.ones(1)))
        self.register_parameter(
            "frozen_weight",
            torch.nn.Parameter(torch.ones(1), requires_grad=False),
        )


def test_training_config_legacy_weight_decay_fills_new_fields() -> None:
    config = TrainingToolConfig(weight_decay=0.02)

    assert config.weight_decay_lora == pytest.approx(0.02)
    assert config.weight_decay_imm == pytest.approx(0.02)


def test_build_optimizer_groups_use_separate_weight_decay() -> None:
    model = DummyTrainableModel()

    groups = build_optimizer_groups(
        model=model,
        lora_lr=2e-4,
        imm_lr=1e-4,
        lora_weight_decay=0.0,
        imm_weight_decay=0.01,
    )

    assert len(groups) == 3
    grouped_params = [{id(param) for param in group["params"]} for group in groups]

    assert groups[0]["weight_decay"] == pytest.approx(0.0)
    assert id(model.lora_adapter_weight) in grouped_params[0]

    assert groups[1]["weight_decay"] == pytest.approx(0.01)
    assert id(model.imm_module_gate) in grouped_params[1]

    assert groups[2]["weight_decay"] == pytest.approx(0.01)
    assert id(model.output_head) in grouped_params[2]


def test_warmup_scheduler_state_can_resume(tmp_path) -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = build_constant_lr_with_warmup_scheduler(
        optimizer=optimizer,
        num_warmup_steps=4,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    for _ in range(2):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    resumed_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    resumed_scheduler = build_constant_lr_with_warmup_scheduler(
        optimizer=resumed_optimizer,
        num_warmup_steps=4,
    )
    load_scheduler_state(checkpoint_dir.as_posix(), resumed_scheduler)

    assert resumed_scheduler.last_epoch == scheduler.last_epoch

    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()

    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
