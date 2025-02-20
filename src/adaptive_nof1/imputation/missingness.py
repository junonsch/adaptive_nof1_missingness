import numpy as np
import random


def insert_missings(length, percentage_missing, mechanism):
    if mechanism == "linear":
        return linear_weighted_sample(length, percentage_missing)
    elif mechanism == "exponential":
        return exponential_weighted_sample(length, percentage_missing)
    elif mechanism == "random":
        return random_sample(length,percentage_missing)
    else:
        print("Not implemented yet.")
        pass


def linear_weighted_sample(length, percentage_missing):
    n_samples = max(1, round((percentage_missing / 100) * length))  # Determine sample size
    numbers = np.arange(1, length + 1)
    weights = numbers / numbers.sum()  # Linear weights
    return np.sort(np.random.choice(numbers, size=n_samples, p=weights, replace=False))

def exponential_weighted_sample(length, percentage_missing, alpha=0.5):
    n_samples = max(1, round((percentage_missing / 100) * length))  # Determine sample size
    numbers = np.arange(1, length + 1)
    weights = np.exp(alpha * numbers)
    weights /= weights.sum()  # Normalize to create probabilities
    return np.sort(np.random.choice(numbers, size=n_samples, p=weights, replace=False))

def random_sample(length,percentage_missing):
    return random.sample(range(length), round(percentage_missing*length))