from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class MemoryDimensionsConfig:
    hidden_dim: int
    key_dim: int
    value_dim: int


@dataclass(frozen=True)
class MemorySlotsConfig:
    session_slots: int = 64
    working_slots: int = 16


@dataclass(frozen=True)
class TurnSummaryConfig:
    pooling_strategy: str = "last_token"
    # "last_token" | "mean_pool" | "attention_pool"
    use_layer_norm: bool = True


@dataclass(frozen=True)
class MemoryControllerConfig:
    # Session memory is long-term turn memory.
    session_merge_gate: float = 1.0
    # Working memory can be disabled for efficiency-focused training.
    use_working_memory: bool = False
    working_merge_gate: float = 1.0


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
    weight_decay: float = 0.01
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


@dataclass(frozen=True)
class InferenceToolConfig:
    memory_enabled: bool = True
    reset_working_memory_per_turn: bool = True
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
