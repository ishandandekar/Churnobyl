"""
This file contains functions and class to run modelling experiments,
tune the hyperparameters and use shap values
"""

import typing as t
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import ensemble, metrics, dummy, neighbors, linear_model, tree, svm
from tqdm.auto import tqdm

# DEV: Add models with names as key-value pair
MODEL_DICT: t.Dict[str, t.Tuple] = {
    "dummy": (
        "dummy",
        dummy.DummyClassifier(strategy="most_frequent"),
    ),
    "knn": ("knn", neighbors.KNeighborsClassifier()),
    "lr": (
        "lr",
        linear_model.LogisticRegression(solver="liblinear", class_weight="balanced"),
    ),
    "svm": ("svm", svm.SVC(kernel="rbf")),
    "rf": ("rf", ensemble.RandomForestClassifier()),
    "gb": (
        "gb",
        ensemble.GradientBoostingClassifier(),
    ),
    "dt": ("dt", tree.DecisionTreeClassifier()),
    "abc": ("abc", ensemble.AdaBoostClassifier()),
    "voting": (
        "voting",
        ensemble.VotingClassifier(estimators=[], voting="soft"),
    ),
    "xgb": ("xgb", xgb.XGBClassifier()),
}


class LearnLab:
    # DEV: Add metrics if you want
    @staticmethod
    def run_experiments(
        model_list: list,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Function to train a whole lot of models sequentially

        Args:
            model_list (list): List of models to train
            X_train (pd.DataFrame): Features on which we want to train the model
            X_test (pd.DataFrame): Features on which we want to test the model
            y_train (pd.DataFrame): Target on which we want to train the model
            y_test (pd.DataFrame): Target on which we want to test the model

        Returns:
            pd.DataFrame: DataFrame that contains the performance results for all the models in `model_list`
        """

        results = dict()
        for name, model in tqdm(model_list):
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
        results = pd.DataFrame.from_dict(results, orient="index")
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
        results = results.sort_values(by=["test_fscore"], ascending=False)
        return results

    # DEV: Add hyperparams for models
    @staticmethod
    def tune_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        n_trials: int,
    ) -> t.Tuple[
        optuna.Study,
        t.Union[ensemble.RandomForestClassifier, xgb.XGBClassifier],
        t.Dict[str, t.Any],
        t.Union[float, np.ndarray],
        t.Union[t.Literal["RandomForest"], t.Literal["XGBoost"]],
    ]:
        """
        Hyperparameter tuning using Optuna
        """

        def tune_random_forest_clf(X_train, X_test, y_train, y_test, n_trials):
            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 100, 1000, step=100
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical(
                        "max_features", [None, "sqrt", "log2"]
                    ),
                    "random_state": 42,
                }
                model = ensemble.RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                auc = metrics.roc_auc_score(y_test, y_pred)
                return auc

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_model = ensemble.RandomForestClassifier(**best_params)
            best_model.fit(X_train, y_train)
            best_preds = best_model.predict(X_test)
            best_metric = metrics.f1_score(y_test, best_preds)
            return study, best_model, best_params, best_metric, "RandomForest"

        def tune_xgboost_classifier(X_train, X_test, y_train, y_test, n_trials):
            def objective(trial):
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "lambda": trial.suggest_float("lambda", 0.01, 10.0, log=True),
                    "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "early_stopping_rounds": 10,
                    "callbacks": [
                        optuna.integration.XGBoostPruningCallback(
                            trial, "validation_0-auc"
                        )
                    ],
                    "random_state": 42,
                }

                model = xgb.XGBClassifier(**params)

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False,
                )

                y_pred = model.predict(X_test)
                auc = metrics.roc_auc_score(y_test, y_pred)
                return auc

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_model = xgb.XGBClassifier(**best_params)

            best_model.fit(X_train, y_train)
            best_preds = best_model.predict(X_test)
            best_metric = metrics.f1_score(y_test, best_preds)
            return study, best_model, best_params, best_metric, "XGBoost"

        (
            rf_study,
            rf_best_model,
            rf_best_params,
            rf_best_metric,
            rf_type_,
        ) = tune_random_forest_clf(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            n_trials=n_trials,
        )
        (
            xgb_study,
            xgb_best_model,
            xgb_best_params,
            xgb_best_metric,
            xgb_type_,
        ) = tune_xgboost_classifier(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            n_trials=n_trials,
        )
        if rf_best_metric > xgb_best_metric:
            return (
                rf_study,
                rf_best_model,
                rf_best_params,
                rf_best_metric,
                rf_type_,
            )
        else:
            return (
                xgb_study,
                xgb_best_model,
                xgb_best_params,
                xgb_best_metric,
                xgb_type_,
            )
