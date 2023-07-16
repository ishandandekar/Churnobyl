import os
from pathlib import Path

import optuna
import pandas as pd
from sklearn import datasets, linear_model, metrics, model_selection

from churnobyl.visualize import Vizard


def test_plot_data_insights():
    data = pd.read_csv("tests/data_for_tests-2.csv", index_col=[0])
    target_dist_path = "target_dist.png"
    contract_dist_path = "contract_dist.png"
    payment_dist_path = "payment_dist.png"
    isp_gender_churn_dist_path = "isp_gender_churn_dist.png"
    partner_churn_dist_path = "partner_churn_dist.png"
    Vizard.plot_data_insights(
        df=data,
        target_dist_path=target_dist_path,
        contract_dist_path=contract_dist_path,
        payment_dist_path=payment_dist_path,
        isp_gender_churn_dist_path=isp_gender_churn_dist_path,
        partner_churn_dist_path=partner_churn_dist_path,
    )
    assert os.path.isfile(target_dist_path) == True
    assert os.path.isfile(contract_dist_path) == True
    assert os.path.isfile(payment_dist_path) == True
    assert os.path.isfile(isp_gender_churn_dist_path) == True
    assert os.path.isfile(partner_churn_dist_path) == True
    os.remove(target_dist_path)
    os.remove(contract_dist_path)
    os.remove(payment_dist_path)
    os.remove(isp_gender_churn_dist_path)
    os.remove(partner_churn_dist_path)


def test_plot_performance_metrics():
    results = pd.read_csv("tests/data_for_tests-1.csv", index_col=[0])
    perf_metrics_path = Path("perf_metrics.png")
    Vizard.plot_performance_metrics(results=results, path=perf_metrics_path)
    assert os.path.isfile(perf_metrics_path) == True
    os.remove(perf_metrics_path)


def test_plot_optuna():
    def objective(trial):
        # Load iris dataset
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define the hyperparameters to tune
        alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)

        # Train the linear regression model with the suggested hyperparameters
        model = linear_model.Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = model.predict(X_val)

        # Calculate the mean squared error
        mse = metrics.mean_squared_error(y_val, y_pred)

        return mse

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    param_importance_path = Path("param_importances.png")
    parallel_coordinate_path = Path("parallel_coordinate.png")
    Vizard.plot_optuna(
        study=study,
        param_importance_path=param_importance_path,
        parallel_coordinate_path=parallel_coordinate_path,
    )
    assert os.path.isfile(param_importance_path) == True
    assert os.path.isfile(parallel_coordinate_path) == True
    os.remove(param_importance_path)
    os.remove(parallel_coordinate_path)


def test_plot_shap():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    shap_path = Path("shap_explainer.png")
    model = linear_model.LinearRegression()
    model.fit(X, y)
    Vizard.plot_shap(model=model, X_train=X, path=shap_path)
    assert os.path.isfile(shap_path) == True
    os.remove(shap_path)
