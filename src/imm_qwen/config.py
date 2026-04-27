from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class MemoryDimensionsConfig:
    key_dim: int
    value_dim: int


@dataclass(frozen=True)
class MemorySlotsConfig:
    session_slots: int = 64


@dataclass(frozen=True)
class TurnSummaryConfig:
    pooling_strategy: str = "last_token"
    # "last_token" | "mean_pool" | "attention_pool"
    use_layer_norm: bool = True


@dataclass(frozen=True)
class MemoryControllerConfig:
    # Session memory is long-term turn memory.
    session_merge_gate: float = 1.0
    # "postnorm": LayerNorm(hidden + IMM_delta), current/default behavior.
    # "prenorm": hidden + LayerNorm(IMM_delta), with inactive memory positions masked out.
    merge_norm_mode: str = "postnorm"

    def __post_init__(self) -> None:
        valid_modes = {"postnorm", "prenorm"}
        if self.merge_norm_mode not in valid_modes:
            raise ValueError(
                "controller.merge_norm_mode must be one of: postnorm, prenorm."
            )


@dataclass(frozen=True)
class ImmPlacementConfig:
    enable_imm: bool = True
    selected_layer_indices: Optional[Tuple[int, ...]] = None
    # If None, top_fraction of layers will be selected.
    top_fraction: float = 0.5


@dataclass(frozen=True)
class ReplacementPolicyConfig:
    policy_name: str = "fifo"
    # Reserved for future policies such as "salience" or "decay"


@dataclass(frozen=True)
class LoraConfigSpec:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class DataSchemaConfig:
    dataset_path: str
    eval_dataset_path: Optional[str] = None
    max_length: int = 2048
    max_history_line_length: int = 256
    max_history_lines: int = 32
    include_history: bool = True
    append_eos_token: bool = True
    # When true, use labels to derive history lookup mask:
    # masked prompt tokens do not read history;
    # unmasked target tokens are allowed to read history.
    derive_history_lookup_mask_from_labels: bool = True


@dataclass(frozen=True)
class ModelBuildConfig:
    model_name_or_path: str
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class TrainingToolConfig:
    learning_rate_lora: float = 2e-4
    learning_rate_imm: float = 2e-4
    weight_decay: Optional[float] = None
    weight_decay_lora: Optional[float] = None
    weight_decay_imm: Optional[float] = None
    warmup_ratio: float = 0.0
    batch_size: int = 4
    num_workers: int = 0
    num_epochs: int = 1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    save_every_steps: int = 0
    output_dir: str = "outputs/imm_qwen"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    eval_every_steps: int = 0
    eval_batch_size: int = 1
    eval_max_new_tokens: int = 128
    eval_temperature: float = 0.0

    def __post_init__(self) -> None:
        resolved_weight_decay = 0.01 if self.weight_decay is None else self.weight_decay
        if self.weight_decay_lora is None:
            object.__setattr__(self, "weight_decay_lora", resolved_weight_decay)
        if self.weight_decay_imm is None:
            object.__setattr__(self, "weight_decay_imm", resolved_weight_decay)

        if self.weight_decay_lora is None or self.weight_decay_lora < 0.0:
            raise ValueError("training.weight_decay_lora must be non-negative.")
        if self.weight_decay_imm is None or self.weight_decay_imm < 0.0:
            raise ValueError("training.weight_decay_imm must be non-negative.")
        if self.weight_decay is not None and self.weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative.")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("training.warmup_ratio must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class InferenceToolConfig:
    memory_enabled: bool = True
    reset_session_memory_on_new_dialog: bool = False
    deterministic: bool = False


@dataclass(frozen=True)
class ImmQwenProjectConfig:
    model: ModelBuildConfig
    memory_dimensions: MemoryDimensionsConfig
    memory_slots: MemorySlotsConfig = field(default_factory=MemorySlotsConfig)
    turn_summary: TurnSummaryConfig = field(default_factory=TurnSummaryConfig)
    controller: MemoryControllerConfig = field(default_factory=MemoryControllerConfig)
    placement: ImmPlacementConfig = field(default_factory=ImmPlacementConfig)
    replacement_policy: ReplacementPolicyConfig = field(default_factory=ReplacementPolicyConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
