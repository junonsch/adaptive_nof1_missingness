import pandas as pd
import numpy as np

def individual_mean_imputation(history, context, action, model):
    ts = context['t']

    if ts == 0:
        fill_value = model.mean[action['treatment']]
    else:
        outcomes = [obs.outcome['outcome'] for obs in history.observations]
        fill_value = np.array(outcomes).mean()
    
    return fill_value


def individual_treatment_mean_imputation(history, context, action, model):
    ts = context['t']
    if ts == 0:
        fill_value = model.mean[action['treatment']]
    else:
        outcomes = [obs.outcome['outcome'] for obs in history.observations if obs.treatment['treatment'] == 1]
        fill_value = np.array(outcomes).mean()
    
    return fill_value