import pandas as pd
from sklearn import datasets, model_selection
import pytest
from churnobyl.model import MODEL_DICT, LearnLab
import optuna


@pytest.fixture
def data():
    # TODO: Load iris binary data
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_run_experiments(data):
    X_train, X_test, y_train, y_test = data
    model_list = list(list(MODEL_DICT.values())[0])
    results = LearnLab.run_experiments(
        model_list=model_list,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == len(model_list)
    assert results.shape[1] == 8
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
