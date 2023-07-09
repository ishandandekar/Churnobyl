from pathlib import Path
import typing as t
import pandas as pd
import xgboost as xgb
from sklearn import (
    model_selection,
    preprocessing,
    dummy,
    metrics,
    ensemble,
    tree,
    neighbors,
    pipeline,
    compose,
    linear_model,
    svm,
)
import optuna as opt
import wandb


# TODO: Add model experiment code
def run_experiments(model_list, metric):
    # TODO: Raise error if best model is not rf or xgb
    ...


# TODO: Add code for rf and xgb tuning
def tune_model():
    ...
