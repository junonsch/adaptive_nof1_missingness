import numpy as np
import torch
import sys
import os
import numpy
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
                "posterior_parameters": obs.posterior_params,
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
        self.rng = numpy.random.default_rng(9001 + patient_id)
        self.mean = mean
        self.variance = variance
        self.patient_id = patient_id

    def multivariate_normal_distribution(debug_data):
        cov = torch.diag_embed(torch.tensor(numpy.sqrt(self.variance)))
        return torch.distributions.MultivariateNormal(torch.tensor(self.mean), cov)

    def generate_context(self, history):
        return {"c": abs(self.rng.normal(0, 0.2))}

    @property
    def additional_config(self):
        return {"expectations_of_interventions": self.mean}

    @property
    def number_of_interventions(self):
        return len(self.mean)

    def observe_outcome(self, action, context):
        treatment_index = action["treatment"]

        outcome = self.rng.normal(
            self.mean[treatment_index], np.sqrt(self.variance[treatment_index])
        )

        return {"outcome": outcome}   

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
                      metrics, 
                      model_mapping,
                      policy_mapping):
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
