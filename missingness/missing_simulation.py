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
number_of_patients = 100
percentage_missing = 0.3
num_patients_missing = 50
missing_mechanism = "linear"
imputation_method_names = ["individual", "global", "locf", "ind_tr",  "global_tr", "knn", "cluster", "dr"] # 
exists = False
results_path = "./results"
random.seed(9001)

treatment_means = [2,2]
effect_parameters_file_ending = f"{treatment_means[0]}{treatment_means[1]}"

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

# Inference Model
inference_model = lambda: NormalKnownVariance(
    prior_mean=0, prior_variance=1, variance=1
)


np.random.seed(9001)
random.seed(9001)


# Initialize the first policy
policy_full = BlockPolicy(
    block_length=block_length,
    internal_policy=ThompsonSampling(
        inference_model=inference_model(),
    ),
)


np.random.seed(9001)
random.seed(9001)

# Initialize the second policy
policy_miss = BlockPolicy(
    block_length=block_length,
    internal_policy=ThompsonSampling(
        inference_model=inference_model(),
    ),
)


study_designs = {
    "n_patients": [number_of_patients],
    "model_from_patient_id": [generating_scenario_II],
    "policy_full": [
       policy_full
    ],
    "policy_miss": [
      policy_miss
    ],
    "pooling": [False],
}


configurations = generate_configuration_cross_product(study_designs)

for imputation_method in imputation_method_names: 

    #### RETURN SIMULATIONS WITH IMPUTATION
    if not exists: 
        print(imputation_method)
        calculated_series  = simulate_missing_configurations(
        configurations, length, percentage_missing, num_patients_missing, missing_mechanism,imputation_method)
       # pd.to_pickle(calculated_series, f"{results_path}/calculated_series_{imputation_method}_{missing_mechanism}_{effect_parameters_file_ending}.pkl")
    else:
        calculated_series = pd.read_pickle(f"{results_path}/calculated_series_{imputation_method}_{missing_mechanism}_{effect_parameters_file_ending}.pkl")
        

    df_result = create_df_result(calculated_series)
    df_result["method"] = imputation_method
    pd.to_pickle(df_result, f"{results_path}/df_result_whatswrong_{imputation_method}_{missing_mechanism}_{effect_parameters_file_ending}.pkl")
    
