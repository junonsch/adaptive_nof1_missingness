import numpy as np
import torch
import sys
import os
from ast import literal_eval
sys.path.append(os.path.abspath("../src"))

from adaptive_nof1 import *
from adaptive_nof1.policies import *
from adaptive_nof1.helpers import *
from adaptive_nof1.inference import *
from adaptive_nof1.metrics import *
import random
import pandas as pd

from adaptive_nof1.missing_series_of_simulations_runner import *
from adaptive_nof1.missing_series_of_simulations_data import *
from adaptive_nof1.series_of_simulations_data import *

from missing_utils import *

# Initial parameters 
block_length = 1
length = 28
number_of_actions = 2
number_of_patients = 1000
percentage_missing = 0.3
num_patients_missing = 500
missing_mechanism = "exponential"
imputation_method_names = ["knn", "ind_tr", "global", "individual", "global_tr", "locf"]
exists = True

random.seed(9001)

treatment_means = [1,0]
effect_parameters_file_ending = f"_{treatment_means[0]}{treatment_means[1]}"

#### SETTINGS

generating_scenario_II = lambda patient_id: NormalModel(
    patient_id, mean=literal_eval(repr(treatment_means)), variance=[1, 1]
)

model_mapping = {
    f"NormalModel(({repr(treatment_means)}, [1, 1]))": "II",
}
policy_mapping = {
    "BlockPolicy(ThompsonSampling(NormalKnownVariance(0, 1, 1)))": "TS",
}

metrics = [
  #  SimpleRegretWithMean(),
 #   CumulativeRegret(),
    KLDivergence(
        data_to_true_distribution=data_to_true_distribution,
        debug_data_to_posterior_distribution=debug_data_to_torch_distribution,
    ),
]


# Inference Model
inference_model = lambda: NormalKnownVariance(
    prior_mean=0, prior_variance=1, variance=1
)

thompson_sampling_policy = BlockPolicy(
    block_length=block_length,
    internal_policy=ThompsonSampling(
        inference_model=inference_model(),
    ),
)

# Full crossover study
study_designs = {
    "n_patients": [number_of_patients],
    "policy": [
        thompson_sampling_policy,
    ],
    "model_from_patient_id": [
        generating_scenario_II,
    ],
    "pooling": [False]
    
}
configurations_ind = generate_configuration_cross_product(study_designs)
configurations_pool = configurations_ind[0]
configurations_pool["pooling"] = True
configurations_pool = [configurations_pool]


for imputation_method in imputation_method_names: 

    #### RETURN SIMULATIONS WITH IMPUTATION
    if not exists: 
        print(imputation_method)
        calculated_series  = simulate_missing_configurations(
        configurations_pool, length, percentage_missing, num_patients_missing, missing_mechanism,imputation_method)
        pd.to_pickle(calculated_series, f"calculated_series_{imputation_method}_{missing_mechanism}_{effect_parameters_file_ending}.pkl")
        
    
    else:
    
        calculated_series = pd.read_pickle(f"calculated_series_{imputation_method}_{missing_mechanism}_{effect_parameters_file_ending}.pkl")
        

    #### RETURN METRICS
    df_metrics = create_metrics_df(calculated_series, imputation_method,metrics, model_mapping, policy_mapping)
    pd.to_pickle(df_metrics, f"df_metrics_{imputation_method}_{missing_mechanism}_{effect_parameters_file_ending}.pkl")


    #### RETURN SCORES FOR FULL OBS
    scores = return_metric_scores(df_metrics,obs="full",method=imputation_method)
    pd.to_pickle(scores_ind_mean, f"scores_{imputation_method}_{missing_mechanism}_{effect_parameters_file_ending}.pkl")

    #### RETURN SCORES FOR MISS OBS
    scores_miss = return_metric_scores(df_metrics,obs="miss",method=f"{imputation_method}_miss")
    pd.to_pickle(scores_miss, f"scores_{imputation_method}_miss_{missing_mechanism}_{effect_parameters_file_ending}.pkl")
