

# Adaptive N-of-1 trials
This repo is based on code and data from Dominik Meier's master thesis on adaptive N-of-1 trials. Please refer to a description of the base code in the github repository.
[![Python Tests](https://github.com/HIAlab/adaptive_nof1/actions/workflows/pytest.yml/badge.svg)](https://github.com/HIAlab/adaptive_nof1/actions/workflows/pytest.yml)
[![Lint](https://github.com/HIAlab/adaptive_nof1/actions/workflows/lint.yml/badge.svg)](https://github.com/HIAlab/adaptive_nof1/actions/workflows/lint.yml)
This repository extends the Bayesian Bandit simulation for adaptive N-of-1 trials by adding simulations for various missingness scenarios and imputation methods.

# Simulation Overview

This repository contains code and results for a simulation study evaluating missing data imputation methods under a Thompson sampling framework.

---

## Simulation Setup

- **Trials:** 1,000 simulated individual trials  
- **Measurements:** 28 per patient  
- **Missing data:** 500 trials include missing values, with **30%** of data points missing  
- **Treatments:** Two treatment arms  
- **Priors:** Conjugate priors `𝒩(0,1)` for treatment effect parameters  
- **Sampler:** Thompson Sampler for treatment assignment and learning  

---

## Scenarios

Two effect scenarios were simulated:

1. **With effect difference** (`Δ = 1`)  
   - Treatment 0 ∼ `𝒩(1,1)`  
   - Treatment 1 ∼ `𝒩(0,1)`

2. **Without effect difference** (`Δ = 0`)  
   - Treatment 0 ∼ `𝒩(2,1)`  
   - Treatment 1 ∼ `𝒩(2,1)`

---

## Missingness Mechanisms

Three missing data mechanisms were implemented:

- **Uniformly random**
- **Linear time-dependent**
- **Exponential time-dependent**

---

## Imputation Methods

Eight imputation methods were compared:

- **Last Observation Carried Forward (LOCF)**
- **Mean imputation** (global / individual, overall / per treatment)
- **k Nearest Neighbors (kNN)** – imputes missing values using the mean outcome of the *k* most similar observations  
- **Clustering-based imputation** – clusters context vectors into *M* groups and imputes missing values using a weighted average of the *m* nearest clusters’ means  
- **Doubly-Robust Estimator** – estimates the coefficient `β` between context `X` and outcome `Y` from complete observations and imputes missing outcomes as  
  `Y_imputed = β * X_obs`

---
