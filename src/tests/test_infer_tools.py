import torch

from imm_qwen.controller import RuleBasedMemoryController
from imm_qwen.infer_tools import InferenceEngine


class DummyTokenizer:
    def __init__(self) -> None:
        self.prompt_ids = torch.tensor([[11, 12, 13]], dtype=torch.long)
        self.generated_ids = torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long)
        self.user_line_ids = torch.tensor([[31, 32]], dtype=torch.long)
        self.assistant_line_ids = torch.tensor([[41, 42]], dtype=torch.long)

    def __call__(self, text, return_tensors="pt"):
        assert return_tensors == "pt"
        if text == "Input:\nhi\n\nAssistant:\n":
            input_ids = self.prompt_ids.clone()
        elif text == "User: hi":
            input_ids = self.user_line_ids.clone()
        elif text == "Assistant: there":
            input_ids = self.assistant_line_ids.clone()
        else:
            raise AssertionError(f"unexpected text: {text}")
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        flat = tokens.tolist()
        if flat == [21, 22]:
            return "there"
        if flat == [11, 12, 13, 21, 22]:
            return "User: hi\nAssistant: there"
        if flat == [11, 12, 13]:
            return "User: hi\nAssistant:"
        raise AssertionError(f"unexpected token sequence: {flat}")


class DummyWrapper:
    def __init__(self) -> None:
        self._collected_keys = []
        self._collected_values = []
        self._collected_valid = []

    def clear_memory_slots(self) -> None:
        self._collected_keys.clear()
        self._collected_values.clear()
        self._collected_valid.clear()

    def append_memory_slot(self, key, value, valid_mask=None) -> None:
        self._collected_keys.append(key)
        self._collected_values.append(value)
        if valid_mask is not None:
            self._collected_valid.append(valid_mask)


class DummyAdapter:
    def __init__(self) -> None:
        self.wrapped_layers = [DummyWrapper()]
        self.prompt_length = None
        self.history_collect_mask = None

    def set_present_query_mode(self, history_lookup_mask=None, prompt_length=0) -> None:
        self.prompt_length = prompt_length

    def set_history_collect_mode(self, attention_mask, row_mask=None) -> None:
        self.history_collect_mask = attention_mask.clone()

    def set_passthrough_mode(self) -> None:
        return None


class DummyModel(torch.nn.Module):
    def __init__(self, adapter: DummyAdapter, generated_ids: torch.Tensor) -> None:
        super().__init__()
        self.adapter = adapter
        self.generated_ids = generated_ids
        self.register_parameter("_dummy_param", torch.nn.Parameter(torch.zeros(1)))
        self.forward_calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        return self.generated_ids.to(input_ids.device)

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.forward_calls.append(
            (
                input_ids.detach().clone(),
                attention_mask.detach().clone(),
            )
        )
        for wrapper in self.adapter.wrapped_layers:
            wrapper._collected_keys.append(torch.tensor([[1.0, 2.0]]))
            wrapper._collected_values.append(torch.tensor([[3.0, 4.0]]))
        return None


def test_inference_writes_completed_turn_to_session_memory(monkeypatch) -> None:
    tokenizer = DummyTokenizer()
    adapter = DummyAdapter()
    model = DummyModel(adapter=adapter, generated_ids=tokenizer.generated_ids)

    monkeypatch.setattr(
        InferenceEngine,
        "_resolve_adapter",
        staticmethod(lambda model: model.adapter),
    )

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        controller=RuleBasedMemoryController(),
    )

    response = engine.generate_response(session_id="s1", user_text="hi", max_new_tokens=8)

    assert response == "there"
    assert adapter.prompt_length == tokenizer.prompt_ids.size(1)
    assert len(model.forward_calls) == 2
    assert torch.equal(model.forward_calls[0][0], tokenizer.user_line_ids)
    assert torch.equal(model.forward_calls[1][0], tokenizer.assistant_line_ids)
    assert torch.equal(
        model.forward_calls[0][1],
        torch.ones_like(tokenizer.user_line_ids),
    )
    assert torch.equal(
        model.forward_calls[1][1],
        torch.ones_like(tokenizer.assistant_line_ids),
    )

    record = engine.session_manager.get_or_create_session("s1", num_layers=1)
    assert record.turn_index == 1
    assert record.layer_banks[0].num_slots == 2
