from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import shap


class Vizard:
    @staticmethod
    def plot_performance_metrics(results: pd.DataFrame, dir: Path) -> None:
        plt.figure(figsize=(10, 6))
        results.plot.barh(xlim=(0, 1), alpha=0.9)
        plt.ylabel("Models")
        plt.xlabel("Metrics")
        plt.title("Model Performance")
        plt.savefig(dir / f"performance_metrics.png")

    @staticmethod
    def plot_optuna(study, dir: Path) -> None:
        # TODO: Add code for making visualizations using optuna
        ...

    @staticmethod
    def plot_shap(model, data, dir: Path) -> None:
        # TODO: Add code for making explainer plots using shap
        ...
