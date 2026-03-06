from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MemoryEfficiencyMetrics:
    recall_score: float
    false_memory_rate: float
    latency_ms: float
    token_budget: int
    memory_slot_usage_ratio: float

    @property
    def recall_per_cost(self) -> float:
        denom = max(self.latency_ms, 1e-6) * max(self.token_budget, 1)
        return self.recall_score / denom


def compute_memory_slot_usage_ratio(valid_slot_count: int, total_slot_count: int) -> float:
    if total_slot_count <= 0:
        return 0.0
    return float(valid_slot_count) / float(total_slot_count)


def compare_recall_per_cost(
    latent_metrics: MemoryEfficiencyMetrics,
    text_prefix_metrics: MemoryEfficiencyMetrics,
) -> Dict[str, float]:
    return {
        "latent_recall_per_cost": latent_metrics.recall_per_cost,
        "text_prefix_recall_per_cost": text_prefix_metrics.recall_per_cost,
        "delta_recall_per_cost": latent_metrics.recall_per_cost - text_prefix_metrics.recall_per_cost,
    }

