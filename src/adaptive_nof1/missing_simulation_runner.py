import copy
import dataclasses
from dataclasses import dataclass
import numpy as np

from adaptive_nof1.basic_types import History, Treatment, Observation
from adaptive_nof1.models.model import Model
from adaptive_nof1.policies.policy import Policy
from adaptive_nof1.missing_simulation_data import MissingSimulationData
#from adaptive_nof1.imputation.mean import individual_mean_imputation
from adaptive_nof1.imputation.imputation import Imputation
import random

@dataclass
class MissingSimulationRunner:
    history: History
    history_miss: History
    policy_full: Policy
    policy_miss: Policy
    model_full: Model
    pooling: bool
    pooledHistory: None = None

    @staticmethod
    def from_model_and_policy_with_copy(model_full: Model, policy_full: Policy,policy_miss: Policy, pooling: bool):
        policy_full.number_of_actions = model_full.number_of_interventions
        policy_miss.number_of_actions = model_full.number_of_interventions
        return MissingSimulationRunner(
            history=History(observations=[]),
            history_miss=History(observations=[]),
            model_full=copy.deepcopy(model_full),
            policy_full =policy_full,
            policy_miss=policy_miss, 
            pooling=pooling,
        )

    @staticmethod
    def from_model_and_policy(model: Model, policy: Policy):
        return MissingSimulationRunner(
            history=History(observations=[]), model=model, policy=policy
        )

    def __str__(self):
        return f"MissingSimulationRunner[{self.policy_full},{self.model_full}]"

    def step(self,missings,model,imputation_method,length):

       
        # for complete track:
        context = self.model_full.generate_context(self.history) ## HEREEEEE
        context["t"] = length

        if context["t"] in missings:
            missing = True
        else:
            missing = False
        any_missings_before = any([miss < context["t"] for miss in missings])
        
        context["patient_id"] = model.patient_id

        hist = History([obs for obs in self.history.observations if obs.context["patient_id"]== context["patient_id"]])
        
        action = self.policy_full.choose_action(hist, context)
        outcome = self.model_full.observe_outcome(action, context)  
        
        # for missing track
        if context["t"] == 0:
            assert not any_missings_before,"There can't be any missings before 0!"

        if not any_missings_before:
            
            self.policy_miss = copy.deepcopy(self.policy_full)
            action_miss = action#self.policy_miss.choose_action(hist, context, divergence_point)
            if not missing:
                outcome_miss = outcome
            else:
                imputer_miss = Imputation(self.history_miss, self.pooledHistory, context, action_miss, model)
                outcome_miss = imputer_miss.impute(imputation_method)
        else:
            action_miss = self.policy_miss.choose_action(self.history_miss, context)
            np.random.seed(9001)
            if missing:
                imputer_miss = Imputation(self.history_miss, self.pooledHistory, context, action_miss, model)
                outcome_miss = imputer_miss.impute(imputation_method)
            else:
               
                outcome_miss = outcome #self.model_full.observe_outcome(action_miss, context)
        if (context["patient_id"] == 60) and (context["t"] in [3,4,5]):
            print(context['t'],action, action_miss)
            print(f"{context['t']} properties of policy_miss: {self.policy_miss.internal_policy.inference.posterior_parameters(2)}")
            print(f"{context['t']} properties of policy_full: {self.policy_full.internal_policy.inference.posterior_parameters(2)}")
            print(context["t"],outcome, outcome_miss)
        if not any_missings_before and not missing:
            assert action == action_miss, f"Mismatch at step {context['t']}: {action} vs {action_miss}"
            assert outcome == outcome_miss, f"Mismatch at step {context['t']}: {outcome} vs {outcome_miss}"

    
        counterfactual_outcomes_miss = [
            self.model_full.observe_outcome(counterfactual_action, context)
            for counterfactual_action in self.policy_miss.available_actions()
        ]
            
        counterfactual_outcomes = [
            self.model_full.observe_outcome(counterfactual_action, context)
            for counterfactual_action in self.policy_full.available_actions()
        ]
        if context["t"] > 0:
            posterior_params_miss = self.policy_miss.internal_policy.inference.posterior_parameters(2)
            posterior_params_full = self.policy_full.internal_policy.inference.posterior_parameters(2)
        else:
            posterior_params_miss = ([self.policy_miss.internal_policy.inference.prior_mean,
                                     self.policy_miss.internal_policy.inference.prior_mean],
                                     [self.policy_miss.internal_policy.inference.prior_variance,
                                     self.policy_miss.internal_policy.inference.prior_variance])
            posterior_params_full = ([self.policy_full.internal_policy.inference.prior_mean,
                                     self.policy_full.internal_policy.inference.prior_mean],
                                     [self.policy_full.internal_policy.inference.prior_variance,
                                     self.policy_full.internal_policy.inference.prior_variance])

        observation = Observation(
            **{
                "patient_id": context["patient_id"],
                "context": context,
                "treatment": action,
                "outcome": outcome,
                "imputation_method": imputation_method,
                "missing": missing,
                "posterior_params": posterior_params_full,
                "counterfactual_outcomes": counterfactual_outcomes,
                "debug_information": self.policy_full.debug_information[-1],
                "debug_data": self.policy_full.debug_data[-1],
                "t": context["t"],#len(self.history),
            }
        )
        observation_miss = Observation(
            **{
                "patient_id": context["patient_id"],
                "context": context,
                "treatment": action_miss,
                "outcome": outcome_miss,
                "imputation_method": imputation_method,
                "missing": missing,
                "posterior_params": posterior_params_miss,
                "counterfactual_outcomes": counterfactual_outcomes_miss,
                "debug_information": self.policy_miss.debug_information[-1],
                "debug_data": self.policy_miss.debug_data[-1],
                "t": context["t"],#len(self.history),
            }
        )
        self.history_miss.add_observation(observation_miss)

        self.history.add_observation(observation)
        return self

    def simulate(self, length) -> MissingSimulationData:
        for _ in range(length):
            self.step()
        return self.get_data()

    def get_data(self):
        return MissingSimulationData(
            history=self.history,
            history_miss=self.history_miss,#.internal_policy.inference.posterior_parameters(2),
            policy_full=self.policy_full,#.internal_policy.inference.posterior_parameters(2),
            policy_miss=self.policy_miss,
            model_full=str(self.model_full),
            patient_id=str(self.model_full.patient_id),
            pooled=self.pooling, #self.pooledHistory != None,
            additional_config={
                **self.model_full.additional_config,
                **self.policy_full.additional_config,
                **self.policy_miss.additional_config,
            },
        )

    def __getitem__(self, index):
        return dataclasses.replace(self, history=self.history[index])
