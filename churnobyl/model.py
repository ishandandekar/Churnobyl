"""
This file contains functions and class to run modelling experiments,
tune the hyperparameters and use shap values
"""
import functools as F
import typing as t
from dataclasses import dataclass
from pathlib import Path

import cloudpickle as cpickle
import multiprocess as mp
import optuna
import polars as pl
import xgboost as xgb
from box import Box
from sklearn import (base, dummy, ensemble, linear_model, metrics, neighbors,
                     svm, tree)

from data import TransformerOutput

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


@dataclass(frozen=True)
class TunerOutput:
    studies: list[optuna.study.Study]
    best_models: list[t.Union[base.BaseEstimator, xgb.XGBClassifier]]
    best_parameters: list[dict]
    best_metrics: list[float]
    names: list[str]
    best_paths: list[Path]

    def __post_init__(self):
        self.table = pl.DataFrame({"models": self.names, "metrics": self.best_metrics})


class LearnLab:
    @staticmethod
    def train_experiments(
        config: Box, transformed_ds: TransformerOutput
    ) -> pl.DataFrame:
        X_train, X_test, y_train, y_test = (
            transformed_ds.X_train,
            transformed_ds.X_test,
            transformed_ds.y_train,
            transformed_ds.y_test,
        )

        def _get_metrics(model, features, labels):
            preds = model.predict(features)
            return (
                metrics.accuracy_score(y_true=labels, y_pred=preds),
                metrics.precision_score(y_true=labels, y_pred=preds),
                metrics.recall_score(y_true=labels, y_pred=preds),
                metrics.f1_score(y_true=labels, y_pred=preds),
            )

        def _trainer(model_item):
            model_name, model_params = (
                list(model_item.keys())[0],
                list(model_item.values())[0].params,
            )
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
            get_metrics = F.partial(_get_metrics, model=model)
            train_accuracy, train_precision, train_recall, train_fscore = get_metrics(
                features=X_train, labels=y_train
            )
            test_accuracy, test_precision, test_recall, test_fscore = get_metrics(
                features=X_test, labels=y_test
            )
            return {
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_fscore": train_fscore,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_fscore": test_fscore,
                "model": model_name,
            }

        with mp.Pool() as pool:
            results: t.List[t.Dict[str, float]] = pool.map(_trainer, config.models)
        return pl.from_dicts(results)

    @staticmethod
    def tune_models(
        config: Box, transformed_ds: t.Type[TransformerOutput], model_dir: Path
    ) -> TunerOutput:
        X_train, X_test, y_train, y_test = (
            transformed_ds.X_train,
            transformed_ds.X_test,
            transformed_ds.y_train,
            transformed_ds.y_test,
        )
        models: list = config.get("models").to_list()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _tuner(model_param_item):
            model_name, model_params, n_trials = (
                list(model_param_item.keys())[0],
                list(model_param_item.values())[0].get("params"),
                list(model_param_item.values())[0].get("n_trials"),
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
                return metrics.f1_score(y_test, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_model = ModelFactory.get(model_name)(**best_params).fit(
                X_train, y_train
            )
            best_metric = study.best_value
            path_ = model_dir / f"{model_name}.pkl"
            with open(path_, "wb") as f_out:
                cpickle.dump(best_model, f_out)
            return study, best_model, best_params, best_metric, model_name, path_

        with mp.Pool() as pool:
            (
                studies,
                best_models,
                best_parameters,
                best_metrics,
                names,
                best_paths,
            ) = zip(*pool.map(_tuner, models))

        (
            sorted_studies,
            sorted_best_models,
            sorted_best_parameters,
            sorted_best_metrics,
            sorted_names,
            sorted_paths,
        ) = zip(
            *sorted(
                zip(
                    studies,
                    best_models,
                    best_parameters,
                    best_metrics,
                    names,
                    best_paths,
                ),
                key=lambda x: x[3],
                reverse=True,
            )
        )
        return TunerOutput(
            studies=sorted_studies,
            best_models=sorted_best_models,
            best_parameters=sorted_best_parameters,
            best_metrics=sorted_best_metrics,
            names=sorted_names,
            best_paths=sorted_paths,
        )
