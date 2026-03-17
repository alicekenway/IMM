import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .config import DataSchemaConfig


@dataclass(frozen=True)
class SupervisedRecord:
    instruction: str
    input: str
    output: str
    system: str
    history: Optional[Any] = None


def load_supervised_records(dataset_path: str) -> List[SupervisedRecord]:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    records: List[SupervisedRecord] = []
    for row in data:
        records.append(
            SupervisedRecord(
                instruction=str(row.get("instruction", "")),
                input=str(row.get("input", "")),
                output=str(row.get("output", "")),
                system=str(row.get("system", "")),
                history=row.get("history"),
            )
        )
    return records


def extract_history_lines(history: Any) -> List[str]:
    if history is None:
        return []
    if isinstance(history, str):
        return [line.strip() for line in history.splitlines() if line.strip()]
    if isinstance(history, list):
        lines: List[str] = []
        for turn in history:
            if isinstance(turn, dict):
                user_text = str(turn.get("user", "")).strip()
                assistant_text = str(turn.get("assistant", "")).strip()
                if user_text:
                    lines.append(f"User: {user_text}")
                if assistant_text:
                    lines.append(f"Assistant: {assistant_text}")
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                if str(turn[0]).strip():
                    lines.append(f"User: {turn[0]}")
                if str(turn[1]).strip():
                    lines.append(f"Assistant: {turn[1]}")
            else:
                text = str(turn).strip()
                if text:
                    lines.append(text)
        return lines
    text = str(history).strip()
    return [text] if text else []


def build_present_turn_prompt_text(record: SupervisedRecord) -> str:
    parts: List[str] = []
    if record.system.strip():
        parts.append(f"System:\n{record.system.strip()}\n")

    if record.instruction.strip():
        parts.append(f"Instruction:\n{record.instruction.strip()}\n")

    present_turn = record.input.strip()
    parts.append(f"Input:\n{present_turn}\n")

    # Keep the assistant prefix in the prompt span so response tokenization
    # does not depend on an injected leading space.
    parts.append("Assistant:\n")
    return "\n".join(parts)


class ImmSupervisedDataset(Dataset):
    """
    LLaMA-Factory-style supervised dataset adapter.

    Output fields for current-turn LM objective:
      - input_ids
      - attention_mask
      - labels
      - history_lookup_mask
      - history_line_input_ids
      - history_line_attention_mask
    """

    def __init__(
        self,
        tokenizer: Any,
        data_config: DataSchemaConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.records = load_supervised_records(data_config.dataset_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        # Present turn is trained as the supervised prediction context.
        prompt_text = build_present_turn_prompt_text(record=record)
        # History lines are encoded separately and used to prefill session memory.
        history_lines = (
            extract_history_lines(record.history)[: self.data_config.max_history_lines]
            if self.data_config.include_history
            else []
        )
        output_text = record.output.strip()
        if self.data_config.append_eos_token and self.tokenizer.eos_token is not None:
            output_text = output_text + self.tokenizer.eos_token

        # Tokenize prompt and output SEPARATELY then concatenate token IDs.
        # This avoids BPE boundary issues where tokenizing them as one string
        # produces different tokens at the junction, causing label mask errors.
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt_text, add_special_tokens=True,
        )
        # Response tokens: no special tokens (BOS already in prompt, EOS in output_text if configured)
        response_ids: List[int] = self.tokenizer.encode(
            output_text, add_special_tokens=False,
        )

        # Truncate to max_length
        max_len = self.data_config.max_length
        total_len = len(prompt_ids) + len(response_ids)
        if total_len > max_len:
            # Keep full prompt, truncate response
            available = max_len - len(prompt_ids)
            if available > 0:
                response_ids = response_ids[:available]
            else:
                prompt_ids = prompt_ids[:max_len]
                response_ids = []

        # Build input_ids and labels from the two separate token lists
        all_ids = prompt_ids + response_ids
        labels_list = [-100] * len(prompt_ids) + response_ids

        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(labels_list, dtype=torch.long)

        # True  -> masked (no history lookup at this token)
        # False -> unmasked (history lookup allowed)
        history_lookup_mask = self._build_history_lookup_mask(labels)

        history_line_input_ids: List[torch.Tensor] = []
        history_line_attention_mask: List[torch.Tensor] = []
        for line_text in history_lines:
            line_encoding = self.tokenizer(
                line_text,
                truncation=True,
                max_length=self.data_config.max_history_line_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
            history_line_input_ids.append(line_encoding["input_ids"].squeeze(0))
            history_line_attention_mask.append(line_encoding["attention_mask"].squeeze(0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "history_lookup_mask": history_lookup_mask,
            "history_line_input_ids": history_line_input_ids,
            "history_line_attention_mask": history_line_attention_mask,
        }

    def _build_history_lookup_mask(self, labels: torch.Tensor) -> torch.Tensor:
        if self.data_config.derive_history_lookup_mask_from_labels:
            # True means masked out (no history lookup). This masks prompt tokens.
            # Unmasked target tokens (labels != -100) are allowed to read history.
            return labels.eq(-100).to(torch.bool)
        # Fallback: allow history lookup everywhere.
        return torch.zeros_like(labels, dtype=torch.bool)


class ImmDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Base supervised tensors are padded by tokenizer utility.
        base_features: List[Dict[str, torch.Tensor]] = []
        labels_batch: List[torch.Tensor] = []
        history_lookup_mask_batch: List[torch.Tensor] = []
        history_line_input_ids_batch: List[List[torch.Tensor]] = []
        history_line_attention_mask_batch: List[List[torch.Tensor]] = []

        for feature in features:
            base_features.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
            )
            labels_batch.append(feature["labels"])
            history_lookup_mask_batch.append(feature["history_lookup_mask"])
            history_line_input_ids_batch.append(feature["history_line_input_ids"])
            history_line_attention_mask_batch.append(feature["history_line_attention_mask"])

        batch = self.tokenizer.pad(
            base_features,
            padding=True,
            return_tensors="pt",
        )

        sequence_length = int(batch["input_ids"].size(1))
        batch_size = len(features)

        labels = torch.full(
            (batch_size, sequence_length),
            fill_value=-100,
            dtype=torch.long,
        )
        history_lookup_mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
        )
        for batch_index, (feature_labels, feature_history_lookup_mask) in enumerate(
            zip(labels_batch, history_lookup_mask_batch)
        ):
            feature_length = int(feature_labels.numel())
            labels[batch_index, :feature_length] = feature_labels
            history_lookup_mask[batch_index, :feature_length] = feature_history_lookup_mask.to(
                torch.bool
            )
        batch["labels"] = labels
        batch["history_lookup_mask"] = history_lookup_mask

        # History lines are padded into [B, H, T_hist] to support batched
        # memory prefill without per-sample Python loops in the train step.
        max_history_lines = max((len(lines) for lines in history_line_input_ids_batch), default=0)
        max_history_length = 1
        for line_list in history_line_input_ids_batch:
            for line_ids in line_list:
                max_history_length = max(max_history_length, int(line_ids.numel()))

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        history_input_ids = torch.full(
            (batch_size, max_history_lines, max_history_length),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        history_attention_mask = torch.zeros(
            (batch_size, max_history_lines, max_history_length),
            dtype=torch.long,
        )
        history_line_mask = torch.zeros((batch_size, max_history_lines), dtype=torch.bool)

        for batch_index, (line_ids_list, line_attn_list) in enumerate(
            zip(history_line_input_ids_batch, history_line_attention_mask_batch)
        ):
            for line_index, (line_ids, line_attn) in enumerate(zip(line_ids_list, line_attn_list)):
                line_length = int(line_ids.numel())
                history_input_ids[batch_index, line_index, :line_length] = line_ids
                history_attention_mask[batch_index, line_index, :line_length] = line_attn
                history_line_mask[batch_index, line_index] = True

        batch["history_input_ids"] = history_input_ids
        batch["history_attention_mask"] = history_attention_mask
        batch["history_line_mask"] = history_line_mask
        return batch
