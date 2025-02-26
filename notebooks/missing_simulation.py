import numpy as np
import torch
from adaptive_nof1 import *
from adaptive_nof1.policies import *
from adaptive_nof1.helpers import *
from adaptive_nof1.inference import *
from adaptive_nof1.metrics import *
import random
import pandas as pd

# Initial parameters
block_length = 1
length = 28
number_of_actions = 2
number_of_patients = 1000
percentage_missing = 0.3
num_patients_missing = 500
missing_mechanism = "linear"
random.seed(9001)

def create_outcome_df(calculated_series, miss=False):

    if miss:
        observations = [patient_data.history_miss.observations for patient_data in calculated_series[0]["result"].simulations]
    else:
        observations = [patient_data.history.observations for patient_data in calculated_series[0]["result"].simulations]
    
    all_obs = []
    for observation in observations:
        # Extract relevant data
        data = [
            {
                "patient_id": obs.patient_id,
                "t": obs.t,
                "treatment": obs.treatment["treatment"],
                "missing": obs.missing,
                "outcome": obs.outcome["outcome"],
              #  "outcome_miss": obs.outcome_miss["outcome_miss"]
            }
            for obs in observation
        ]
        
        # Create DataFrame
        df_obs = pd.DataFrame(data)
        all_obs.append(df_obs)
    
    all_obs = pd.concat(all_obs)

    return all_obs

def create_df_result(calculated_series):
    all_obs = create_outcome_df(calculated_series, miss=False)
    all_obs_miss = create_outcome_df(calculated_series, miss=True)
    df_result = pd.merge(all_obs,all_obs_miss, on=["patient_id","t"],suffixes=["","_miss"])

    return df_result


# Scenarios
class NormalModel(Model):
    def __init__(self, patient_id, mean, variance):
        self.rng = numpy.random.default_rng(patient_id)
        self.mean = mean
        self.variance = variance
        self.patient_id = patient_id

    def multivariate_normal_distribution(debug_data):
        cov = torch.diag_embed(torch.tensor(numpy.sqrt(self.variance)))
        return torch.distributions.MultivariateNormal(torch.tensor(self.mean), cov)

    def generate_context(self, history):
        return {}

    @property
    def additional_config(self):
        return {"expectations_of_interventions": self.mean}

    @property
    def number_of_interventions(self):
        return len(self.mean)

    def observe_outcome(self, action, context):
        treatment_index = action["treatment"]
        return {
            "outcome": self.rng.normal(
                self.mean[treatment_index], numpy.sqrt(self.variance[treatment_index])
            )
        }        

    def __str__(self):
        return f"NormalModel({self.mean, self.variance})"

def debug_data_to_torch_distribution(debug_data):
    mean = debug_data["mean"]
    # + the true variance of 1
    variance = numpy.array(debug_data["variance"]) + 1
    cov = torch.diag_embed(torch.tensor(numpy.sqrt(variance)))
    return torch.distributions.MultivariateNormal(torch.tensor(mean), cov)


def data_to_true_distribution(data):
    mean = data.additional_config["expectations_of_interventions"]
    cov = torch.eye(len(mean))
    return torch.distributions.MultivariateNormal(torch.tensor(mean), cov)

def create_metrics_df(calculated_series, method, 
                      metrics=metrics, 
                      model_mapping=model_mapping,
                      policy_mapping=policy_mapping):
    df = SeriesOfSimulationsData.score_data(
        [s["result"] for s in calculated_series],
        metrics,
        {"model": lambda x: model_mapping[x], "policy": lambda x: policy_mapping[x]},
    )
    df["obs"] = "full"
    df_miss = SeriesOfMissingSimulationsData.score_missing_data(
        [s["result"] for s in calculated_series],
        metrics,
        {"model": lambda x: model_mapping[x], "policy": lambda x: policy_mapping[x]},
    )
    df_miss["obs"] = "miss"
    df_metrics = pd.concat([df,df_miss],axis=0)
    df_metrics["method"] = method

    return df_metrics

def return_metric_scores(df_metrics,obs,method):
    filtered_df1 = df_metrics[(df_metrics["obs"]==obs)].reset_index().copy()
    filtered_df = filtered_df1.loc[filtered_df1["t"] == filtered_df1["t"].max()].reset_index()
    
    groupby_columns = ["model", "policy"]
    pivoted_df = filtered_df.pivot(
        index=["model", "policy", "simulation", "patient_id"],
        columns="metric",
        values="score",
    )
    table = pivoted_df.groupby(groupby_columns).agg(["mean", "std"])
    
    policy_ordering = ["Fixed", "ETC", "SH", "UCB", "TS", "Pooled UCB", "Pooled TS"]
    
    # Convert the 'policy' column in the MultiIndex to a Categorical type with the specified order
    table = table.reset_index()
    table["policy"] = pd.Categorical(
        table["policy"], categories=policy_ordering, ordered=True
    )
    
    # Sort the DataFrame first by 'model' then by the now-ordered 'policy'
    sorted_table = table.sort_values(by=["model", "policy"]).set_index(groupby_columns)
    sorted_table["method"] = method
    return sorted_table


generating_scenario_II = lambda patient_id: NormalModel(
    patient_id, mean=[1, 0], variance=[1, 1]
)

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



print("knn")
calculated_series_knn  = simulate_missing_configurations(
configurations_pool, length, percentage_missing, num_patients_missing, missing_mechanism,"knn")
pd.to_pickle(calculated_series_knn, f"calculated_series_knn_{missing_mechanism}.pkl")

print("individual_treatment_mean")
calculated_series_individual_tr_mean  = simulate_missing_configurations(
    configurations_ind, length, percentage_missing, num_patients_missing, missing_mechanism, "individual_treatment_mean"
)
pd.to_pickle(calculated_series_individual_tr_mean, f"calculated_series_ind_tr_{missing_mechanism}.pkl")

print("global_mean")
calculated_series_global_mean  = simulate_missing_configurations(
    configurations_pool, length, percentage_missing, num_patients_missing, missing_mechanism, "global_mean"
)
pd.to_pickle(calculated_series_global_mean, f"calculated_series_global_{missing_mechanism}.pkl")

print("individual_mean")
calculated_series_individual_mean  = simulate_missing_configurations(
configurations_ind, length, percentage_missing, num_patients_missing, missing_mechanism, "individual_mean")
pd.to_pickle(calculated_series_individual_mean, f"calculated_series_ind_{missing_mechanism}.pkl")

print("global_treatment_mean")
calculated_series_global_tr_mean  = simulate_missing_configurations(
    configurations_pool, length, percentage_missing, num_patients_missing, missing_mechanism,"global_treatment_mean"
)
pd.to_pickle(calculated_series_global_tr_mean, f"calculated_series_global_tr_{missing_mechanism}.pkl")

print("locf")
calculated_series_locf  = simulate_missing_configurations(
    configurations_ind, length, percentage_missing, num_patients_missing, missing_mechanism, "locf"
)
pd.to_pickle(calculated_series_locf, f"calculated_series_locf_{missing_mechanism}.pkl")


metrics = [
  #  SimpleRegretWithMean(),
 #   CumulativeRegret(),
    KLDivergence(
        data_to_true_distribution=data_to_true_distribution,
        debug_data_to_posterior_distribution=debug_data_to_torch_distribution,
    ),
]
model_mapping = {
    "NormalModel(([1, 0], [1, 1]))": "II",
}
policy_mapping = {
    "BlockPolicy(ThompsonSampling(NormalKnownVariance(0, 1, 1)))": "TS",
}


df_metrics_ind_mean = create_metrics_df(calculated_series_individual_mean, "individual_mean")
pd.to_pickle(df_metrics_ind_mean, f"df_metrics_ind_mean_{missing_mechanism}.pkl")

df_metrics_glob_mean = create_metrics_df(calculated_series_global_mean, "global_mean")
pd.to_pickle(df_metrics_glob_mean, f"df_metrics_glob_mean_{missing_mechanism}.pkl")

df_metrics_ind_tr_mean = create_metrics_df(calculated_series_individual_tr_mean, "individual_tr_mean")
pd.to_pickle(df_metrics_ind_tr_mean, f"df_metrics_ind_tr_mean_{missing_mechanism}.pkl")

df_metrics_glob_tr_mean = create_metrics_df(calculated_series_individual_tr_mean, "global_tr_mean")
pd.to_pickle(df_metrics_glob_tr_mean, f"df_metrics_glob_tr_mean_{missing_mechanism}.pkl")

df_metrics_knn_mean = create_metrics_df(calculated_series_knn, "knn")
pd.to_pickle(df_metrics_knn_mean, f"df_metrics_knn_mean_{missing_mechanism}.pkl")

df_metrics_locf_mean = create_metrics_df(calculated_series_locf, "locf")
pd.to_pickle(df_metrics_ind_mean, f"df_metrics_ind_mean_{missing_mechanism}.pkl")

scores_ind_mean = return_metric_scores(df_metrics_ind_mean,obs="full",method="ind_mean")
pd.to_pickle(scores_ind_mean, f"scores_ind_mean_{missing_mechanism}.pkl")
scores_glob_mean = return_metric_scores(df_metrics_ind_mean,obs="full",method="glob_mean")
pd.to_pickle(scores_glob_mean, f"scores_glob_mean_{missing_mechanism}.pkl")

scores_ind_tr_mean = return_metric_scores(df_metrics_ind_tr_mean,obs="full",method="ind_tr_mean")
pd.to_pickle(scores_ind_tr_mean, f"scores_ind_tr_mean_{missing_mechanism}.pkl")

scores_glob_tr_mean = return_metric_scores(df_metrics_ind_tr_mean,obs="full",method="glob_tr_mean")
pd.to_pickle(scores_glob_tr_mean, f"scores_glob_tr_mean_{missing_mechanism}.pkl")

scores_knn_mean = return_metric_scores(df_metrics_knn_mean,obs="full",method="knn")
pd.to_pickle(scores_knn_mean, f"scores_locf_mean_{missing_mechanism}.pkl")

scores_locf_mean = return_metric_scores(df_metrics_locf_mean,obs="full",method="locf")
pd.to_pickle(scores_locf_mean, f"scores_locf_mean_{missing_mechanism}.pkl")