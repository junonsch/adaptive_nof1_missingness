from adaptive_nof1.basic_types import History

from dataclasses import dataclass, field


@dataclass
class MissingSimulationData:
    history: History
    history_miss: History
    model: str
    policy: str
    patient_id: int
    pooled: bool = False
    additional_config: dict = field(default_factory=dict)

    def __str__(self):
        return f"SimulationData"

    @property
    def configuration(self):
        return {
            "policy": self.policy,
            "model": self.model,
            "patient_id": self.patient_id,
            "pooled": self.pooled,
            **self.additional_config,
        }
