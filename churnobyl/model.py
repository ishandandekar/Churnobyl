"""
This file contains functions and class to run modelling experiments,
tune the hyperparameters and use shap values
"""

import pickle as pkl
import typing as t
from pathlib import Path

import optuna
import pandas as pd
import shap
import wandb
import xgboost as xgb
from sklearn import (
    compose,
    dummy,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    neighbors,
    pipeline,
    preprocessing,
    svm,
    tree,
)


class LearnLab:
    # TODO: Add model experiment code
    @staticmethod
    def run_experiments(model_list, X_train, X_test, y_train, y_test, model_dir: Path):
        results = dict()
        for name, model in model_list:
            model.fit(X_train, y_train)
            path_ = model_dir / f"{name}.pkl"
            with open(path_, "wb") as f:
                pkl.dump(model, f)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_accuracy = metrics.accuracy_score(y_train, train_predictions)
            (
                train_precision,
                train_recall,
                train_fscore,
                _,
            ) = metrics.precision_recall_fscore_support(y_train, train_predictions)
            test_accuracy = metrics.accuracy_score(y_train, train_predictions)
            (
                test_precision,
                test_recall,
                test_fscore,
                _,
            ) = metrics.precision_recall_fscore_support(y_train, test_predictions)
            results[name] = (
                train_accuracy,
                train_precision,
                train_recall,
                train_fscore,
                test_accuracy,
                test_precision,
                test_recall,
                test_fscore,
            )
        results = pd.DataFrame.from_dict(results)
        results.columns = [
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_fscore",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_fscore",
        ]
        results = results.sort_values(by=["test_fcore"], ascending=False)
        return results

    # TODO: Add code for rf and xgb tuning
    @staticmethod
    def tune_model(X_train, X_test, y_train, y_test) -> optuna.Study:
        ...
