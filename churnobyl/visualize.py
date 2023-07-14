from pathlib import Path
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import shap


class Vizard:
    @staticmethod
    def plot_data_insights(df: pd.DataFrame) -> None:
        ...

    @staticmethod
    def plot_performance_metrics(results: pd.DataFrame, path: Path) -> None:
        if not str(path).endswith(".png"):
            raise Exception(f"{path} should be a path to `.png`")
        plt.figure(figsize=(10, 6))
        results.plot.barh(xlim=(0, 1), alpha=0.9)
        plt.ylabel("Models")
        plt.xlabel("Metrics")
        plt.title("Model Performance")
        plt.savefig(path, format="png")
        plt.close()

    @staticmethod
    def plot_optuna(
        study: optuna.Study, param_importance_path: Path, parallel_coordinate_path: Path
    ) -> None:
        if not str(param_importance_path).endswith(".png"):
            raise Exception(f"{param_importance_path} should be a path to `.png`")
        if not str(parallel_coordinate_path).endswith(".png"):
            raise Exception(f"{parallel_coordinate_path} should be a path to `.png`")
        param_importances = optuna.visualization.plot_param_importance(study)
        param_importances.write_image(param_importance_path)

        parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
        parallel_coordinate.write_image(parallel_coordinate_path)

    @staticmethod
    def plot_shap(model, X_train: pd.DataFrame, path: Path) -> None:
        if not str(path).endswith(".png"):
            raise Exception(f"{path} should be a path to `.png`")
        explainer: shap.Explainer = shap.Explainer(model)
        features_names = X_train.columns.to_list()
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, features_names=features_names)
        plt.savefig(path, format="png")
        plt.close()
