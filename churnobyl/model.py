"""
This file contains functions and class to run modelling experiments,
tune the hyperparameters and use shap values
"""

from pathlib import Path
import typing as t
from dataclasses import dataclass
from box import Box, BoxList
import optuna
import pandas as pd
import xgboost as xgb
from sklearn import ensemble, metrics, dummy, neighbors, linear_model, tree, svm, base
from tqdm.auto import tqdm

# DEV: Add models with names as key-value pair
ModelFactory: t.Dict[str, t.Union[xgb.XGBClassifier, base.BaseEstimator]] = {
    "dummy": dummy.DummyClassifier,
    "knn": neighbors.KNeighborsClassifier,
    "lr": linear_model.LogisticRegression,
    "svm": svm.SVC,
    "rf": ensemble.RandomForestClassifier,
    "gb": ensemble.GradientBoostingClassifier,
    "dt": tree.DecisionTreeClassifier,
    "abc": ensemble.AdaBoostClassifier,
    "voting": ensemble.VotingClassifier,
    "xgb": xgb.XGBClassifier,
    "ext": ensemble.ExtraTreesClassifier,
    "sgd": linear_model.SGDClassifier,
}


@dataclass
class TunerOutput:
    studies: list[optuna.study.Study]
    best_models: list[t.Union[base.BaseEstimator, xgb.XGBClassifier]]
    best_parameters: list[dict]
    best_metrics: list[float]
    names: list[str]

@dataclass
class ModelEngineOutput:
    results: pd.DataFrame
    tuner: TunerOutput
    paths: list[Path]


class LearnLab:
    @staticmethod
    def train_experiments(
        config: Box,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> pd.DataFrame:
        results = dict()
        for idx in config.models:
            model_name, model_params = list(idx.keys())[0], list(idx.values())[0].params
            if model_name == "voting":
                estimators_list = list()
                for estimator in model_params.estimators:
                    model_, model_params_ = (
                        list(estimator.keys())[0],
                        list(estimator.values())[0].params,
                    )
                    model__ = ModelFactory.get(model_)
                    model__ = model__(**model_params_)
                    estimators_list.append((model_, model__))
                model_params["estimators"] = estimators_list
            model = ModelFactory.get(model_name)
            model = model(**model_params)
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_accuracy = metrics.accuracy_score
            train_precision = metrics.precision_score(
                y_true=y_train, y_pred=train_predictions
            )
            train_recall = metrics.recall_score(
                y_true=y_train, y_pred=train_predictions
            )
            train_fscore = metrics.f1_score(y_true=y_train, y_pred=train_predictions)
            test_accuracy = metrics.accuracy_score(y_test, test_predictions)
            test_precision = metrics.precision_score(
                y_true=y_test, y_pred=test_predictions
            )
            test_recall = metrics.recall_score(y_true=y_test, y_pred=test_predictions)
            test_fscore = metrics.f1_score(y_true=y_test, y_pred=test_predictions)
            results[model_name] = (
                train_accuracy,
                train_precision,
                train_recall,
                train_fscore,
                test_accuracy,
                test_precision,
                test_recall,
                test_fscore,
            )
        results: pd.DataFrame = pd.DataFrame.from_dict(
            results,
            orient="index",
            columns=[
                "train_accuracy",
                "train_precision",
                "train_recall",
                "train_fscore",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_fscore",
            ],
        ).sort_values(by=["test_fscore"], ascending=False)

        return results

    @staticmethod
    def tune_models(
        config: Box,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> TunerOutput:
        studies, best_models, best_parameters, best_metrics, names = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )
        models: BoxList = config.get("models").to_list()
        for model_param_item in models:
            model_name, model_params = (
                list(model_param_item.keys())[0],
                list(model_param_item.values())[0].get("params"),
            )

            def objective(trial: optuna.Trial):
                params = dict()
                for k, v in model_params.items():
                    args = v.get("args")
                    args["name"] = k

                    if v.get("strategy") == "float":
                        params[k] = trial.suggest_float(**args)
                    elif v.get("strategy") == "int":
                        params[k] = trial.suggest_int(**args)
                    elif v.get("strategy") == "cat":
                        params[k] = trial.suggest_categorical(**args)
                model = ModelFactory.get(model_name)
                model = model(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                return (metrics.f1_score(y_test, preds),)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=config.get("n_trials"))
            best_params = study.best_params
            best_model = ModelFactory.get(model_name)(**best_params).fit(
                X_train, y_train
            )
            best_metric = study.best_value
            studies.append(study)
            best_models.append(best_model)
            best_parameters.append(best_params)
            best_metrics.append(best_metric)
            names.append(model_name)
        sorted_studies, sorted_best_models, sorted_best_parameters, sorted_best_metrics, sorted_names = zip(*sorted(zip(studies, best_metrics, best_parameters, best_metrics, names), key=lambda x: x[3], reverse=True)))
        return TunerOutput(
            studies=sorted_studies,
            best_models=sorted_best_models,
            best_parameters=sorted_best_parameters,
            best_metrics=sorted_best_metrics,
            names=sorted_names,
        )
