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


@dataclass
class MissingSeriesOfSimulationsRunner:
    simulations: List[MissingSimulationRunner]
    #pooling: bool = False

    def __init__(
        self,
        model_from_patient_id: Callable[[int], Model],
        n_patients: int,
        policy,
        pooling=False,
    ):
        self.simulations = [
            MissingSimulationRunner.from_model_and_policy_with_copy(
                model_from_patient_id(index),
                policy,
                pooling
            )
            for index in range(n_patients)
        ]
        assert all_equal(
            [str(s.policy) for s in self.simulations]
        ), "Not all policies are the same. Usually, you need to set __str__() somewhere"
        assert all_equal(
            [str(s.model) for s in self.simulations]
        ), "Not all models are the same. Usually, you need to set __str__() somewhere"

        self.n_patients = n_patients
        self.model_from_patient_id = model_from_patient_id
        self.pooling = pooling

    def simulate(self, length, percentage_missing, num_patients_missing,missing_mechanism,imputation_method) -> SeriesOfMissingSimulationsData:
        
        patients_missing= np.sort(random.sample(range(len(self.simulations)), num_patients_missing))
        positions_missing = {p:insert_missings(length, percentage_missing, missing_mechanism) for p in patients_missing}
       # positions_missing = random.sample(range(length), round(percentage_missing*length))
        print(f"These patients have missing values: {patients_missing}")
        for i in progressbar(range(length)):
            print(f"AT TIME POINT {i}")
            for num_sim, simulation in enumerate(self.simulations): # n_patients
                model = self.model_from_patient_id(num_sim)
                if num_sim in patients_missing:
                    missings = positions_missing[num_sim]
                else: 
                    missings = []
                simulation.pooling = self.pooling
                if self.pooling:
                    
                    histories = [simulation.history for simulation in self.simulations]
                    pooled_history = History.fromListOfHistories(histories)
                    for simulation in self.simulations:
                        simulation.pooledHistory = pooled_history
                    simulation.step(missings,model,imputation_method,i)
                else:
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
            "policy": str(self.simulations[0].policy),
            "model": str(self.simulations[0].model),
            "pooling": self.pooling,
        }


def simulate_missing_configurations(configurations, length, percentage_missing, num_patients_missing,missing_mechanism,imputation_method):
    calculated_series = []
    for configuration in configurations:
        result = MissingSeriesOfSimulationsRunner(**configuration).simulate(length, percentage_missing,num_patients_missing,missing_mechanism,imputation_method)

        calculated_series.append(
            {"configuration": result.configuration, "result": result}
        )

    # config_to_simulation_data = {
    #     str(simulation.configuration): simulation
    #     for d in calculated_series
    #     for simulation in d["result"].simulations
    # }
    return calculated_series#, config_to_simulation_data
