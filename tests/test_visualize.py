from churnobyl.visualize import Vizard
import pandas as pd
from pathlib import Path
import os
import optuna
from sklearn import linear_model, datasets
import pandera as pa


def test_plot_data_insights():
    schema = pa.DataFrameSchema().from_yaml("../data_schemas/training_schema.yaml")
    synthetic_data = schema.example(size=10)
    target_dist_path = "target_dist.png"
    demographic_dist_path = "demographic_dist.png"
    cust_acc_dist_path = "cust_acc_dist.png"
    Vizard.plot_data_insights(
        df=synthetic_data,
        target_dist_path=target_dist_path,
        demographic_dist_path=demographic_dist_path,
        cust_acc_dist_path=cust_acc_dist_path,
    )
    assert os.path.isfile(target_dist_path) == True
    assert os.path.isfile(demographic_dist_path) == True
    assert os.path.isfile(cust_acc_dist_path) == True
    os.remove(target_dist_path)
    os.remove(demographic_dist_path)
    os.remove(cust_acc_dist_path)
    assert ...


def test_plot_performance_metrics():
    results = pd.read_csv("./data_for_tests.csv", index_col=[0])
    perf_metrics_path = Path("perf_metrics.png")
    Vizard.plot_performance_metrics(results=results, path=perf_metrics_path)
    assert os.path.isfile(perf_metrics_path) == True
    os.remove(perf_metrics_path)


def test_plot_optuna():
    study = optuna.create_study()
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
    model = linear_model.LinearRegression().fit(X, y)
    Vizard.plot_shap(model=model, X_train=X, path=shap_path)
    assert os.path.isfile(shap_path) == True
    os.remove(shap_path)
