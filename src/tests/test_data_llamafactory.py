import torch

from imm_qwen.data_llamafactory import ImmDataCollator


class DummyTokenizer:
    pad_token_id = 0

    def pad(self, features, padding=True, return_tensors="pt"):
        assert padding is True
        assert return_tensors == "pt"
        max_len = max(feature["input_ids"].numel() for feature in features)
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for idx, feature in enumerate(features):
            length = feature["input_ids"].numel()
            input_ids[idx, :length] = feature["input_ids"]
            attention_mask[idx, :length] = feature["attention_mask"]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collator_keeps_one_padded_history_slot_when_batch_has_no_history() -> None:
    collator = ImmDataCollator(tokenizer=DummyTokenizer())
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
            "history_lookup_mask": torch.tensor([True, False, False], dtype=torch.bool),
            "history_line_input_ids": [],
            "history_line_attention_mask": [],
        },
        {
            "input_ids": torch.tensor([4, 5], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, 5], dtype=torch.long),
            "history_lookup_mask": torch.tensor([True, False], dtype=torch.bool),
            "history_line_input_ids": [],
            "history_line_attention_mask": [],
        },
    ]

    batch = collator(features)

    assert batch["history_input_ids"].shape == (2, 1, 1)
    assert batch["history_attention_mask"].shape == (2, 1, 1)
    assert batch["history_line_mask"].shape == (2, 1)
    assert not batch["history_line_mask"].any()
