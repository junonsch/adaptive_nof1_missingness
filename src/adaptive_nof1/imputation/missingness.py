import numpy as np
import random

def insert_missings(length, percentage_missing, mechanism, patient_id):
    np.random.seed(patient_id)
    if mechanism == "linear":
        return linear_weighted_sample(length, percentage_missing, patient_id)
    elif mechanism == "exponential":
        return exponential_weighted_sample(length, percentage_missing, patient_id)
    elif mechanism == "random":
        return random_sample(length,percentage_missing, patient_id)
    else:
        print("Not implemented yet.")
        pass


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