import numpy as np
import random

def insert_missings(length, percentage_missing, mechanism, patient_id):
   
    random.seed(patient_id)
    if mechanism == "linear":
        return linear_weighted_sample(length, percentage_missing, patient_id)
    elif mechanism == "exponential":
        return exponential_weighted_sample(length, percentage_missing, patient_id)
    elif mechanism == "random":
        return random_sample(length,percentage_missing, patient_id)
    else:
        print("Not implemented yet.")
        pass


def set_missingness_positions(simulations, num_patients_missing, trial_length, percentage_missing, missing_mechanism):
    patients_missing= np.sort(random.sample(range(len(simulations)), num_patients_missing))
    patients_missing = [int(p) for p in patients_missing]
    positions_missing = {p:insert_missings(trial_length, percentage_missing, missing_mechanism, p) for p in patients_missing}
    
    return patients_missing, positions_missing

def linear_weighted_sample(length, percentage_missing, patient_id):
    n_samples = max(1, round(percentage_missing * length))  # Determine sample size
    numbers = np.arange(1, length+1)
    weights = numbers / numbers.sum()  # Linear weights
    misses = np.sort(np.random.choice(numbers, size=n_samples, p=weights, replace=False))
    misses = misses - 1
    return misses

def exponential_weighted_sample(length, percentage_missing, patient_id, alpha=0.5):
    n_samples = max(1, round(percentage_missing * length))  # Determine sample size
    numbers = np.arange(1, length + 1)
    weights = np.exp(alpha * numbers)
    weights /= weights.sum()  # Normalize to create probabilities
    misses = np.sort(np.random.choice(numbers, size=n_samples, p=weights, replace=False))
    misses = misses - 1
    return misses

def random_sample(length,percentage_missing, patient_id):
    misses = np.sort(random.sample(range(length), round(percentage_missing*length)))
    return misses