import pandas as pd
from sklearn import datasets, model_selection
import pytest
from churnobyl.model import ModelFactory, LearnLab
import optuna
from sklearn.base import BaseEstimator


@pytest.fixture
def data():
    iris = datasets.load_iris()

    X, y = iris.data, iris.target
    class_to_remove = 2

    # Find indices of samples belonging to the class to remove
    indices_to_remove = y == class_to_remove

    # Remove the samples of the chosen class
    X_filtered, y_filtered = X[~indices_to_remove], y[~indices_to_remove]

    # Split the filtered data into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_run_experiments(data):
    X_train, X_test, y_train, y_test = data
    model_list = list(list(ModelFactory.values())[:2])
    results = LearnLab.run_experiments(
        model_list=model_list,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == len(model_list)
    assert results.columns.tolist() == [
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_fscore",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_fscore",
    ]


def test_tune_model(data):
    X_train, X_test, y_train, y_test = data
    (
        study,
        best_model,
        best_params,
        best_metric,
        best_type_,
    ) = LearnLab.tune_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, n_trials=1
    )
    assert isinstance(study, optuna.study.Study)
    assert isinstance(best_params, dict)
    assert isinstance(best_metric, float)
    assert isinstance(best_type_, str)


def test_modelFactory():
    keys = list(ModelFactory.keys())
    model_idx = [tup[0] for tup in ModelFactory.values()]
    models = [tup[1] for tup in ModelFactory.values()]
    assert keys == model_idx
    for model in models:
        assert isinstance(model, BaseEstimator)
