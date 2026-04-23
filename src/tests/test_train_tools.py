import pytest
import torch
from types import SimpleNamespace

from imm_qwen.config import TrainingToolConfig
from imm_qwen.train_tools import (
    _unwrap_distributed_model,
    build_constant_lr_with_warmup_scheduler,
    build_optimizer_groups,
    load_scheduler_state,
    run_validation,
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


class DummyValidationAdapter:
    def clear_all_memory_slots(self) -> None:
        return None

    def set_present_query_mode(self, history_lookup_mask=None, prompt_length=0) -> None:
        return None

    def set_passthrough_mode(self) -> None:
        return None


class DummyValidationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.tensor(
            [[[0.1, 1.0, 0.0], [0.2, 0.1, 0.9], [0.0, 0.5, 0.2]]],
            dtype=torch.float32,
        )

    def forward(self, *args, **kwargs):
        return SimpleNamespace(logits=self.logits)

    def generate(self, input_ids, attention_mask, pad_token_id, **kwargs):
        del attention_mask, pad_token_id, kwargs
        generated_suffix = torch.tensor([[2]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, generated_suffix], dim=1)


class DummyDistributedWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        raise AssertionError("run_validation should use the unwrapped model, not the DDP wrapper")


class DummyValidationTokenizer:
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        flat = token_ids.tolist()
        return " ".join(str(token) for token in flat)


def test_unwrap_distributed_model_returns_inner_module() -> None:
    inner = DummyValidationModel()
    wrapped = DummyDistributedWrapper(inner)

    assert _unwrap_distributed_model(wrapped) is inner


def test_run_validation_uses_unwrapped_model_for_generate(monkeypatch) -> None:
    inner_model = DummyValidationModel()
    wrapped_model = DummyDistributedWrapper(inner_model)
    adapter = DummyValidationAdapter()
    tokenizer = DummyValidationTokenizer()
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 2]], dtype=torch.long),
        "history_lookup_mask": torch.tensor([[True, False, False]], dtype=torch.bool),
        "history_input_ids": torch.zeros((1, 0, 1), dtype=torch.long),
        "history_attention_mask": torch.zeros((1, 0, 1), dtype=torch.long),
        "history_line_mask": torch.zeros((1, 0), dtype=torch.bool),
    }

    monkeypatch.setattr(
        "imm_qwen.train_tools.resolve_imm_adapter",
        lambda model: adapter,
    )

    avg_loss = run_validation(
        model=wrapped_model,
        val_dataloader=[batch],
        tokenizer=tokenizer,
        training_config=TrainingToolConfig(),
        global_step=10,
        device=torch.device("cpu"),
        log_fn=lambda *_args, **_kwargs: None,
    )

    assert avg_loss >= 0.0
