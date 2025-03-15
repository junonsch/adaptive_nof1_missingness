from adaptive_nof1.basic_types import History
from adaptive_nof1.policies import Policy

from dataclasses import dataclass, field


@dataclass
class MissingSimulationData:
    history: History
    history_miss: History
    model_full: str
    policy_full: str
    policy_miss: str
    patient_id: int
    pooled: bool = False
    additional_config: dict = field(default_factory=dict)

    def __str__(self):
        return f"SimulationData"

    @property
    def configuration(self):
        return {
            "policy_full": self.policy_full,
            "policy_miss": self.policy_miss,
            "model_full": self.model_full,
            "patient_id": self.patient_id,
            "pooled": self.pooled,
            **self.additional_config,
        }
