import typing as t
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import (
    ensemble,
    metrics,
    dummy,
    neighbors,
    linear_model,
    tree,
    svm,
    model_selection,
)
from sklearn.datasets import load_iris

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
        for name, model in model_list:
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


def main():
    model_list = [list(MODEL_DICT.values())[0]]
    print(model_list)
    X, y = load_iris(return_X_y=True, as_frame=True)
    X, y = X[:100], y[:100]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results = LearnLab.run_experiments(
        model_list=model_list,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
