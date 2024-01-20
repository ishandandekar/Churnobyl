"""
This file contains functions and class to run modelling experiments,
tune the hyperparameters and use shap values
"""
import functools as F
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import cloudpickle as cpickle
import multiprocess as mp
import numpy as np
import optuna
from scipy.sparse import spmatrix
import polars as pl
import xgboost as xgb
from box import Box
from sklearn import base, dummy, ensemble, linear_model, metrics, neighbors, svm, tree
from src.data import TransformerOutput
from src.exceptions import ConfigValidationError

# DEV: Add models with names as key-value pair
# Dictionary collections of supported models
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


# Class to encapsulate output of the tuning the models
@dataclass()
class TunerOutput:
    studies: list[optuna.study.Study]
    best_models: list[t.Union[base.BaseEstimator, xgb.XGBClassifier]]
    best_parameters: list[dict]
    best_metrics: list[float]
    names: list[str]
    best_paths: list[Path]
    table: pl.DataFrame = field(init=False)

    def __post_init__(self):
        (
            self.studies,
            self.best_models,
            self.best_parameters,
            self.best_metrics,
            self.names,
            self.paths,
        ) = zip(
            *sorted(
                zip(
                    self.studies,
                    self.best_models,
                    self.best_parameters,
                    self.best_metrics,
                    self.names,
                    self.best_paths,
                ),
                key=lambda x: x[3],
                reverse=True,
            )
        )
        # Save insights as `.csv`
        self.table = pl.DataFrame({"Models": self.names, "Metrics": self.best_metrics})
        dir_path = self.best_paths[0].parent
        self.table.write_csv(dir_path / "tuning_results.csv")


class LearnLab:
    @staticmethod
    def train_experiments(
        config: Box, transformed_ds: TransformerOutput
    ) -> pl.DataFrame:
        """
        Runs training experiments from a plethora of models specified

        Args:
            config (Box): Configuration mapping for training
            transformed_ds (TransformerOutput): Contains training and test features and labels

        Returns:
            pl.DataFrame: Contains metrics and name of the model as a DataFrame
        """
        X_train, X_test, y_train, y_test = (
            transformed_ds.X_train,
            transformed_ds.X_test,
            transformed_ds.y_train,
            transformed_ds.y_test,
        )

        def _get_metrics(
            model: t.Union[base.BaseEstimator, xgb.XGBClassifier],
            features: t.Union[np.ndarray, spmatrix],
            labels: t.Union[np.ndarray, spmatrix],
        ) -> t.Tuple[float, float, float, float]:
            """
            Returns accuracy, precision, recall and f1score

            Args:
                model (t.Union[base.BaseEstimator, xgb.XGBClassifier]): Model to get the metrics for
                features (t.Union[np.ndarray, spmatrix]): Features to predict for
                labels (t.Union[np.ndarray, spmatrix]): True labels for the features

            Returns:
                t.Tuple[float, float, float, float]: Accuracy, Precision, Recall, F1-score
            """
            preds = model.predict(features)
            return (
                metrics.accuracy_score(y_true=labels, y_pred=preds),
                metrics.precision_score(y_true=labels, y_pred=preds),
                metrics.recall_score(y_true=labels, y_pred=preds),
                metrics.f1_score(y_true=labels, y_pred=preds),
            )

        def _trainer(model_item) -> t.Dict[str, float]:
            """
            Traines over one model

            Returns:
                t.Dict[str, float]: Key-value pair of train and test metrics with the name of the model
            """
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

        # Flag to utilize multiprocessing
        if config.multiprocess:
            with mp.Pool() as pool:
                results: t.List[t.Dict[str, float]] = pool.map(_trainer, config.models)
        else:
            results: t.List[t.Dict[str, float]] = list()
            for item in config.models:
                results.append(_trainer(item))
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

        def _tuner(
            model_param_item,
        ) -> t.Tuple[optuna.study.Study, t.Any, t.Dict[str, t.Any], float, str, Path]:
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

        if config.multiprocess:
            with mp.Pool() as pool:
                (
                    studies,
                    best_models,
                    best_parameters,
                    best_metrics,
                    names,
                    best_paths,
                ) = zip(*pool.map(_tuner, models))
        else:
            studies, best_models, best_parameters, best_metrics, names, best_paths = (
                list(),
                list(),
                list(),
                list(),
                list(),
                list(),
            )
            for model in models:
                (
                    study,
                    best_model,
                    best_parameter,
                    best_metric,
                    name,
                    best_path,
                ) = _tuner(model)
                studies.append(study)
                best_models.append(best_model)
                best_parameters.append(best_parameter)
                best_metrics.append(best_metric)
                names.append(name)
                best_paths.append(best_path)
        return TunerOutput(
            studies=studies,
            best_models=best_models,
            best_parameters=best_parameters,
            best_metrics=best_metrics,
            names=names,
            best_paths=best_paths,
        )


def validate_model_config(config) -> None:
    model_config = config.model

    # Validate train args
    models = model_config.train.to_dict().get("models")
    model_list = [list(item.keys())[0] for item in models]
    model_factory_keys = sorted(list(ModelFactory.keys()))
    for model_name in model_list:
        if model_name not in model_factory_keys:
            raise ConfigValidationError(
                f"{model_name} not available. Choose one of the specified models {model_factory_keys}"
            )
    del models
    del model_list
    del model_name

    # Validate tuner args
    tune_args = model_config.tune.to_dict()
    tune_models = tune_args.get("models")
    for item in tune_models:
        k, v = list(item.keys())[0], list(item.values())[0]
        if k not in model_factory_keys:
            raise ConfigValidationError(
                f"Model {k} not available. Choose one of {model_factory_keys}"
            )
        if not v.get("n_trials") and not isinstance(v.get("n_trials"), int):
            raise ConfigValidationError(
                f"Model {k} does not have a valid `n_trials` attribute. Make sure to keep it as an integer."
            )
    del k, v, tune_args, tune_models, item, model_factory_keys
