from adaptive_nof1.basic_types import History
import numpy as np
import pandas as pd

class Imputation:

    def __init__(
        self,
        history: History,
        context: int,
        action,
        model,
    ):
        self.history = history 
        self.context = context
        self.action = action 
        self.model = model

    def impute(self, imputation_method):
        if imputation_method == "locf":
            outcome_miss = {"outcome": self.locf(),
                            "imputation_method":imputation_method}
        elif imputation_method == "individual_mean":
            outcome_miss = {"outcome": self.individual_mean_imputation(),
                            "imputation_method":imputation_method}
        elif imputation_method == "individual_treatment_mean":
            outcome_miss = {"outcome": self.individual_treatment_mean_imputation(),
                            "imputation_method":imputation_method}
        elif imputation_method == "global_mean":
            outcome_miss = {"outcome": self.global_mean_imputation(),
                            "imputation_method":imputation_method}
        elif imputation_method == "global_treatment_mean":
            outcome_miss = {"outcome": self.global_treatment_mean_imputation(),
                            "imputation_method":imputation_method}
        elif imputation_method == "knn":
            outcome_miss = {"outcome": self.knn_imputation(),
                            "imputation_method":imputation_method}
        elif imputation_method == "cluster":
            outcome_miss = {"outcome": self.cluster_imputation(),
                            "imputation_method":imputation_method}
        elif imputation_method == "all":
            print("Not implemented yet.")
            pass
            # for method in ["individual_mean", "individual_treatment_mean", "global_mean", "global_treatment_mean", "knn"]:#, "cluster"]:
            #     outcomes_miss.append({"outcome":self.impute(method),
            #                           "imputation_method":imputation_method})
        return outcome_miss
            
    def locf(self):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            observations = [obs for obs in self.history.observations if obs.patient_id == self.model.patient_id]
            fill_value = observations[-1].outcome["outcome"]
            print(f"chosen value for locf is: {fill_value}")
            print(observations[-1])
        return fill_value

    def individual_mean_imputation(self):
        ts = self.context['t']
       
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            outcomes = [obs.outcome['outcome'] for obs in self.history.observations]
            fill_value = np.array(outcomes).mean()
        
        return fill_value


    def individual_treatment_mean_imputation(self):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            outcomes = [obs.outcome['outcome'] for obs in self.history.observations if obs.treatment['treatment'] == self.action['treatment']]
            fill_value = np.array(outcomes).mean()
        
        return fill_value
    
    def global_mean_imputation(self):
        return self.individual_mean_imputation()


    def global_treatment_mean_imputation(self):
        return self.individual_treatment_mean_imputation()

    def knn_imputation(self,k=3):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            vectors = {}
            comp_vec = {}
            comp_vec[self.context["patient_id"]] = [(obs.context['t'], obs.treatment['treatment'], obs.outcome['outcome']) for obs in self.history.observations if obs.context["patient_id"] == self.context["patient_id"] and obs.context["t"] != self.context["t"] ]
            patient_ids = [obs.context["patient_id"] for obs in self.history.observations]
            for patient_id in pd.Series(patient_ids).unique():
                vectors[patient_id] = [(obs.context['t'], obs.treatment['treatment'], obs.outcome['outcome']) for obs in self.history.observations if not obs.missing and obs.context["patient_id"] != self.context["patient_id"] and obs.context["patient_id"] == patient_id and obs.context["t"] != self.context["t"] ]
            # Function to extract the third element from each tuple and return as a NumPy array
            def extract_vector(data):
                return np.array([tup[2] for tup in data if len(tup) > 2])

            # Extract the reference vector
            ref_vec = extract_vector(comp_vec[list(comp_vec.keys())[0]])

            # Initialize a list to store distances
            distances = []

            # Calculate the Euclidean distance between ref_vec and each vector in vectors
            for key, value in vectors.items():
                vec_key = extract_vector(value)
                # Ensure vectors are of the same length
                if len(ref_vec) == len(vec_key):
                    dist = np.linalg.norm(ref_vec - vec_key)
                    distances.append((key, dist))

            # Sort distances by the second item in each tuple (the distance)
            distances.sort(key=lambda x: x[1])

            k_closest_keys = [key for key, dist in distances[:k]]
            # Extract the last elements from each tuple in the k closest vectors
            last_elements = []
            for key in k_closest_keys:
                last_elements.extend([tup[2] for tup in vectors[key] if len(tup) > 2])

            # Calculate the mean of the last elements
            if last_elements:
                fill_value = np.mean(last_elements)  
            else:
                pass
        return fill_value

    def cluster_imputation(self):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            pass


