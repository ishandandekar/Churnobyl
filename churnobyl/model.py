"""
This file contains functions and class to run modelling experiments,
tune the hyperparameters and use shap values
"""

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
import shap


class LearnLab:
    # TODO: Add model experiment code
    # TODO: Don't forget to add Ridge and Lasso regression
    @staticmethod
    def run_experiments(model_list, X_train, X_test, y_train, y_test):
        # TODO: Raise error if best model is not rf or xgb
        results = dict()
        for name, model in model_list:
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_metric = metrics.accuracy_score(train_predictions, y_train)
            test_metric = metrics.accuracy_score(test_predictions, y_test)
            results[name] = (test_metric, train_metric)
        results = pd.DataFrame.from_dict(results)
        results.columns = ["test_metric", "train_metric"]
        results = results.sort_values(by=["test_metric"], ascending=False)
        return results

    # TODO: Add code for rf and xgb tuning
    @staticmethod
    def tune_model():
        ...

    # TODO: Add shap code
    @staticmethod
    def explain_model():
        ...
