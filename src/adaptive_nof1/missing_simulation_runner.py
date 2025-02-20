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


@dataclass
class MissingSimulationRunner:
    history: History
    history_miss: History
    policy: Policy
    model: Model
    pooling: bool
    pooledHistory: None = None

    @staticmethod
    def from_model_and_policy_with_copy(model: Model, policy: Policy,pooling: bool):
        policy.number_of_actions = model.number_of_interventions
        return MissingSimulationRunner(
            history=History(observations=[]),
            history_miss=History(observations=[]),
            model=copy.deepcopy(model),
            policy=copy.deepcopy(policy),
            pooling=pooling,
        )

    @staticmethod
    def from_model_and_policy(model: Model, policy: Policy):
        return MissingSimulationRunner(
            history=History(observations=[]), model=model, policy=policy
        )

    def __str__(self):
        return f"MissingSimulationRunner[{self.policy},{self.model}]"

    def step(self,missings,model,imputation_method,length):
        if self.policy.is_stopped:
            return self


        # for complete track:
        context = model.generate_context(self.history)
        context["t"] = length
        if len(missings) > 0:
            print(f"missing values are here: {missings}")
        if context["t"] in missings:
            missing = True
        else:
            missing = False
        any_missings_before = any([miss < context["t"] for miss in missings])
        context["patient_id"] = model.patient_id
        print(f"PATIENT ID IS {context['patient_id']}")

        complete_action = self.policy.choose_action(self.history, context)
        if self.pooling:
            len_pooled_hist =  [obs for obs in self.pooledHistory.observations if obs.context["patient_id"]== context["patient_id"]]
            if (len(len_pooled_hist) >0) and any_missings_before:
                action = self.policy.choose_action(self.pooledHistory, context)
            else: 
                action = complete_action
        else:
            action = complete_action
            #action = self.policy.choose_action(self.history, context)
        
        outcome = self.model.observe_outcome(action, context)  


        # for missing track
        if self.pooling and not any_missings_before:
            history_miss = self.pooledHistory
        else:
            history_miss = self.history_miss


        if missing:
            action_miss = self.policy.choose_action(history_miss, context)
        else: 
            action_miss = complete_action
        
       # action_miss = self.policy.choose_action(history_miss, context)

        
        imputer_miss = Imputation(history_miss, context, action_miss, model)
        if missing:
            #outcome_miss = {"outcome": individual_mean_imputation(history_miss, context, action_miss, model)}
            outcome_miss = imputer_miss.impute(imputation_method)
            outcome_miss["imputation_method"] = imputation_method
        else:
            outcome_miss = self.model.observe_outcome(action_miss, context)  
            outcome_miss["imputation_method"] = imputation_method
            if not any_missings_before:
                outcome_miss = outcome
                outcome_miss["imputation_method"] = imputation_method

        
        counterfactual_outcomes_miss = [
            self.model.observe_outcome(counterfactual_action, context)
            for counterfactual_action in self.policy.available_actions()
        ]
         
        
            
        counterfactual_outcomes = [
            self.model.observe_outcome(counterfactual_action, context)
            for counterfactual_action in self.policy.available_actions()
        ]


        observation = Observation(
            **{
                "patient_id": context["patient_id"],
                "context": context,
                "treatment": action,
                "outcome": outcome,
                "imputation_method": imputation_method,
                "missing": missing,
                "counterfactual_outcomes": counterfactual_outcomes,
                "debug_information": self.policy.debug_information[-1],
                "debug_data": self.policy.debug_data[-1],
                "t": context["t"],#len(self.history),
            }
        )
        observation_miss = Observation(
            **{
                "patient_id": context["patient_id"],
                "context": context,
                "treatment": action_miss,
                "outcome": outcome_miss,
                "imputation_method": outcome_miss["imputation_method"],
                "missing": missing,
                "counterfactual_outcomes": counterfactual_outcomes_miss,
                "debug_information": self.policy.debug_information[-1],
                "debug_data": self.policy.debug_data[-1],
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
            history_miss=self.history_miss,
            policy=str(self.policy),
            model=str(self.model),
            patient_id=str(self.model.patient_id),
            pooled=self.pooledHistory != None,
            additional_config={
                **self.model.additional_config,
                **self.policy.additional_config,
            },
        )

    def __getitem__(self, index):
        return dataclasses.replace(self, history=self.history[index])
