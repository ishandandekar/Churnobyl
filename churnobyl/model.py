"""
This file contains functions and class to run modelling experiments,
tune the hyperparameters and use shap values
"""
import functools as F
import typing as t
from dataclasses import dataclass
from pathlib import Path

import cloudpickle as cpickle
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


@dataclass
class TunerOutput:
    studies: list[optuna.study.Study]
    best_models: list[t.Union[base.BaseEstimator, xgb.XGBClassifier]]
    best_parameters: list[dict]
    best_metrics: list[float]
    names: list[str]
    best_paths: list[Path]


def _get_metrics(data, model, y_true):
    preds = model.predict(data)
    return (
        metrics.accuracy_score(y_true=y_true, y_pred=preds),
        metrics.precision_score(y_true=y_true, y_pred=preds),
        metrics.recall_score(y_true=y_true, y_pred=preds),
        metrics.f1_score(y_true=y_true, y_pred=preds),
    )


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
        results: t.Dict[
            str, t.Tuple[float, float, float, float, float, float, float, float]
        ] = dict()
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
            get_metrics = F.partial(_get_metrics, model=model)
            train_accuracy, train_precision, train_recall, train_fscore = get_metrics(
                data=X_train, y_true=y_train
            )
            test_accuracy, test_precision, test_recall, test_fscore = get_metrics(
                data=X_test, y_true=y_test
            )
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
        return (
            pl.DataFrame(results)
            .transpose()
            .lazy()
            .with_columns(
                pl.col("column_0").alias("train_accuracy"),
                pl.col("column_1").alias("train_precision"),
                pl.col("column_2").alias("train_recall"),
                pl.col("column_3").alias("train_fscore"),
                pl.col("column_4").alias("test_accuracy"),
                pl.col("column_5").alias("test_precision"),
                pl.col("column_6").alias("test_recall"),
                pl.col("column_7").alias("test_fscore"),
            )
            .select(
                pl.col("train_accuracy"),
                pl.col("train_precision"),
                pl.col("train_recall"),
                pl.col("train_fscore"),
                pl.col("test_accuracy"),
                pl.col("test_precision"),
                pl.col("test_recall"),
                pl.col("test_fscore"),
            )
            .sort("test_fscore", descending=True)
            .collect()
        )

    @staticmethod
    def tune_models(
        config: Box, transformed_ds: t.Type[TransformerOutput], model_dir: Path
    ) -> TunerOutput:
        studies, best_models, best_parameters, best_metrics, names, best_paths = (
            list(),
            list(),
            list(),
            list(),
            list(),
            list(),
        )
        X_train, X_test, y_train, y_test = (
            transformed_ds.X_train,
            transformed_ds.X_test,
            transformed_ds.y_train,
            transformed_ds.y_test,
        )
        models: list = config.get("models").to_list()
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

            path_ = model_dir / model_name
            with open(path_, "wb") as f_out:
                cpickle.dump(best_model, f_out)
            studies.append(study)
            best_models.append(best_model)
            best_parameters.append(best_params)
            best_metrics.append(best_metric)
            names.append(model_name)
            best_paths.append(path_)
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
                    best_metrics,
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
