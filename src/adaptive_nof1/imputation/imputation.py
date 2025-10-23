from adaptive_nof1.basic_types import History
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import random

class Imputation:

    def __init__(
        self,
        history: History,
        pooled_history: History,
        context: int,
        action,
        model,
    ):
        self.history = history 
        self.pooled_history = pooled_history 
        self.context = context
        self.action = action 
        self.model = model

    def impute(self, imputation_method):
        random.seed(9001)
        if imputation_method == "locf":
            outcome_miss = {"outcome": self.locf()}
        elif imputation_method == "individual":
            outcome_miss = {"outcome": self.individual_mean_imputation()}
        elif imputation_method == "ind_tr":
            outcome_miss = {"outcome": self.individual_treatment_mean_imputation()}
        elif imputation_method == "global":
            outcome_miss = {"outcome": self.global_mean_imputation()}
        elif imputation_method == "global_tr":
            outcome_miss = {"outcome": self.global_treatment_mean_imputation()}
        elif imputation_method == "knn":
            outcome_miss = {"outcome": self.knn_imputation()}
        elif imputation_method == "cluster":
            outcome_miss = {"outcome": self.cluster_imputation()}
        elif imputation_method == "dr":
            outcome_miss = {"outcome": self.DR_imputation()}
        elif imputation_method == "amelia":
            # not implemented
            outcome_miss = {"outcome": self.amelia_imputation()}
        elif imputation_method == "all":
            print("Not implemented yet.")
            pass
        return outcome_miss
            
    def locf(self):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            observations = [obs for obs in self.history.observations if obs.patient_id == self.model.patient_id]
            fill_value = observations[-1].outcome["outcome"]
           
        return fill_value

    def individual_mean_imputation(self):
        ts = self.context['t']
       
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            outcomes = [obs.outcome['outcome'] for obs in self.history.observations if obs.patient_id == self.model.patient_id]
            fill_value = np.array(outcomes).mean()
        
        return fill_value


    def individual_treatment_mean_imputation(self):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            outcomes = [obs.outcome['outcome'] for obs in self.history.observations if obs.treatment['treatment'] == self.action['treatment'] and obs.patient_id == self.model.patient_id]
            fill_value = np.array(outcomes).mean()
        
        return fill_value
    
    def global_mean_imputation(self):
        ts = self.context['t']
       
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            outcomes = [obs.outcome['outcome'] for obs in self.pooled_history.observations]
            fill_value = np.array(outcomes).mean()
        
        return fill_value


    def global_treatment_mean_imputation(self):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            outcomes = [obs.outcome['outcome'] for obs in self.pooled_history.observations if obs.treatment['treatment'] == self.action['treatment']]
            fill_value = np.array(outcomes).mean()
        
        return fill_value

    def prep_vectors_and_comp_vec(self):
        comp_vec = {}
        comp_vec[self.context["patient_id"]] = [(obs.context['t'], obs.treatment['treatment'], obs.outcome['outcome']) for obs in self.pooled_history.observations if obs.context["patient_id"] == self.context["patient_id"] and obs.context["t"] != self.context["t"] ]
        vectors = {}
        patient_ids = [obs.context["patient_id"] for obs in self.pooled_history.observations]
        for patient_id in pd.Series(patient_ids).unique():
            vectors[patient_id] = [(obs.context['t'], obs.treatment['treatment'], obs.outcome['outcome']) for obs in self.pooled_history.observations if not obs.missing and obs.context["patient_id"] != self.context["patient_id"] and obs.context["patient_id"] == patient_id and obs.context["t"] != self.context["t"] ]

        return vectors, comp_vec

    def fit_estimate_beta(self, X, targets, lamb): ## lamb is the regulization parameter
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        lambI = np.eye(X_b.shape[1]) * lamb
    
        beta_DR = np.linalg.inv(X_b.T.dot(X_b) + lambI ).dot(X_b.T).dot(targets)
        
        return beta_DR
    
    def predict(self, beta_DR, X):
        """
        Predict target values for given input features X.
        X: numpy array of shape (n_samples, n_features)
        """
        return beta_DR[0] + X.dot(beta_DR[1:])



    def DR_imputation(self): 

        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            vectors, comp_vec = self.prep_vectors_and_comp_vec()
    
            X = []
            targets = []  # Store target values (third element)
            keys = []
            
            for key, value in vectors.items():
                if value:  # Ignore empty lists
                    feature_vector = value[0][:2]  
                    target_value = value[0][2]  
                    X.append(feature_vector)
                    targets.append(target_value)
                    keys.append(key)
            
            X = np.array(X)
            targets = np.array(targets)
        
            beta_DR = self.fit_estimate_beta(X, targets,lamb=1)
    
            comparator_vector = np.array(list(comp_vec.values())[0][0][:2])
            
            fill_value = self.predict(beta_DR, comparator_vector)
        return fill_value

    def cluster_imputation(self):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            vectors, comp_vec = self.prep_vectors_and_comp_vec()

            X = []
            targets = []  # Store target values (third element)
            keys = []
            
            for key, value in vectors.items():
                if value:  # Ignore empty lists
                    feature_vector = value[0][:2]  
                    target_value = value[0][2]  
                    X.append(feature_vector)
                    targets.append(target_value)
                    keys.append(key)
            
            X = np.array(X)
            targets = np.array(targets)
            
            n_clusters = min(len(X), 3)  # Choose number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            cluster_means = {}
            for cluster in range(n_clusters):
                cluster_means[cluster] = np.mean(targets[labels == cluster])
            
            comparator_vector = np.array(list(comp_vec.values())[0][0][:2])  
            closest_cluster = min(range(n_clusters), key=lambda c: euclidean(comparator_vector, kmeans.cluster_centers_[c]))
            
            fill_value = cluster_means[closest_cluster]
            
            return fill_value



    def knn_imputation(self,k=3):
        ts = self.context['t']
        if ts == 0:
            fill_value = self.model.mean[self.action['treatment']]
        else:
            vectors = {}
            comp_vec = {}
            comp_vec[self.context["patient_id"]] = [(obs.context['t'], obs.treatment['treatment'], obs.outcome['outcome']) for obs in self.pooled_history.observations if obs.context["patient_id"] == self.context["patient_id"] and obs.context["t"] != self.context["t"] ]
            patient_ids = [obs.context["patient_id"] for obs in self.pooled_history.observations]
            for patient_id in pd.Series(patient_ids).unique():
                vectors[patient_id] = [(obs.context['t'], obs.treatment['treatment'], obs.outcome['outcome']) for obs in self.pooled_history.observations if not obs.missing and obs.context["patient_id"] != self.context["patient_id"] and obs.context["patient_id"] == patient_id and obs.context["t"] != self.context["t"] ]
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

    def amelia_imputation(self):
        # iin progress

        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        # import R's utils package
        utils = rpackages.importr('utils')

        # R package names
        packnames = [('Amelia')]

        # Selectively install what needs to be installed.
        names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        Amelia = importr('Amelia')

        dfs = []
        for obs in self.pooled_history.observations:
            context= obs.context
            treatment= obs.treatment
            outcome= obs.outcome
            missing = obs.missing
            imputation_method = obs.imputation_method
            combined = {**context, **treatment, **outcome, 'missing': missing, 'imputation_method': imputation_method}
            dfs.append(combined)
            # Create a one-row DataFrame
           # df = pd.DataFrame([combined])
           # dfs.append(df)
        dfs[-1].update({"t":self.context["t"],
                        "patient_id":self.context["patient_id"],
                        "treatment":self.action["treatment"],
                        "missing":True})
        complete_df = pd.DataFrame(dfs)

        with localconverter(ro.default_converter + pandas2ri.converter):
            dataframe = ro.conversion.py2rpy(complete_df.reset_index(drop=True))
       
        imputed = Amelia.amelia(x = dataframe,
                    m = 1,
                    p2s = 1,
                    idvars = StrVector(['patient_id']), # if Pandas df with an index then use 'row.names'
                    #noms = StrVector(["coly"]),
                    ords = "colz",
                    ts = "t",
                    cs = None,
                    empri = 0.05 * len(dataframe),
                    polytime = 3,
                    splinetime = 9,
                    intercs = True,
                    bounds = np.array([[10, 0, 1000], [11, 0, 1000], [12, 0, 1000]]), # the specific dimensions of the dataframe
                    parallel = 'multicore',
                    ncpus = 8) # cpus on computer


