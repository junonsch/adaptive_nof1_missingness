from __future__ import annotations

from adaptive_nof1.missing_series_of_simulations_data import SeriesOfMissingSimulationsData
from adaptive_nof1.metrics.metric import score_missing_df
from adaptive_nof1.models.model import Model
from adaptive_nof1.missing_simulation_runner import MissingSimulationRunner
#from adaptive_nof1.simulation_runner import SimulationRunner
from adaptive_nof1.helpers import all_equal
from adaptive_nof1.imputation.missingness import insert_missings

from adaptive_nof1.basic_types import History
import random

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict

from tqdm.auto import tqdm as progressbar
import seaborn as sns
import panel
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import copy

@dataclass
class MissingSeriesOfSimulationsRunner:
    simulations: List[MissingSimulationRunner]
    #pooling: bool = False

    def __init__(
        self,
        model_from_patient_id: Callable[[int], Model],
        n_patients: int,
        policy_full,
        policy_miss,
        pooling=False,

    ):
        np.random.seed(9001)  # Reset before creating policies
        
        # Ensure both policies are initialized from the same random state
        policy_full_copy = copy.deepcopy(policy_full)
        policy_miss_copy = copy.deepcopy(policy_miss)
        self.simulations = [
            MissingSimulationRunner.from_model_and_policy_with_copy(
                model_from_patient_id(index),
                policy_full_copy,
                policy_miss_copy, 
                pooling,
            )
            for index in range(n_patients)
        ]
        assert all_equal(
            [str(s.policy_full) for s in self.simulations]
        ), "Not all policies are the same. Usually, you need to set __str__() somewhere"
        assert all_equal(
            [str(s.model_full) for s in self.simulations]
        ), "Not all models are the same. Usually, you need to set __str__() somewhere"

        self.n_patients = n_patients
        self.model_from_patient_id = model_from_patient_id
        self.pooling = pooling

    def simulate(self, length, percentage_missing, num_patients_missing,missing_mechanism,imputation_method) -> SeriesOfMissingSimulationsData:
        random.seed(9001)
      #  np.random.seed(9001)
        if imputation_method in ["individual", "ind_tr", "locf"]:
            self.pooling = False
        else:
            self.pooling = True
        patients_missing= np.sort(random.sample(range(len(self.simulations)), num_patients_missing))
        positions_missing = {p:insert_missings(length, percentage_missing, missing_mechanism, p) for p in patients_missing}
        for i in progressbar(range(length)):
            #print(f"AT TIME POINT {i}")
            for num_sim, simulation in enumerate(self.simulations): # n_patients
              #  print(f"This patient {num_sim}:")
                model = self.model_from_patient_id(num_sim)
                if num_sim in patients_missing:
                    missings = positions_missing[num_sim]
                else: 
                    missings = []
                simulation.pooling = self.pooling
                if self.pooling:                    
                    histories = [simulation.history_miss for simulation in self.simulations]
                    pooled_history = History.fromListOfHistories(histories)
                    simulation.pooledHistory = pooled_history
                simulation.step(missings,model,imputation_method,i)
                

        return SeriesOfMissingSimulationsData(
            simulations=[simulation.get_data() for simulation in self.simulations],
            configuration=self.configuration,
            )

    def clone_with_policy(self, new_policy):
        runner = MissingSeriesOfSimulationsRunner(
            model_from_patient_id=self.model_from_patient_id,
            n_patients=self.n_patients,
            policy=new_policy,
        )
        return runner

    @property
    def configuration(self):
        return {
            "policy_full": self.simulations[0].policy_full.internal_policy.inference.posterior_parameters(2),#str(self.simulations[0].policy_full),
            "policy_miss": self.simulations[0].policy_miss.internal_policy.inference.posterior_parameters(2),#str(self.simulations[0].policy_miss),
            "model_full": str(self.simulations[0].model_full),
            "pooling": self.pooling,
        }


def simulate_missing_configurations(configurations, length, percentage_missing, num_patients_missing,missing_mechanism,imputation_method):
    calculated_series = []
    np.random.seed(9001)  
    for configuration in configurations:
        result = MissingSeriesOfSimulationsRunner(**configuration).simulate(length, percentage_missing,num_patients_missing,missing_mechanism,imputation_method)

        calculated_series.append(
            {"configuration": result.configuration, "result": result}
        )

    return calculated_series
