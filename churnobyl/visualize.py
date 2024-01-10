"""
Contains functions to make plots and visualizations
"""
import typing as t
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl
import seaborn as sns
from box import Box

warnings.simplefilter("ignore")
plt.switch_backend("agg")
plt.ioff()


class Vizard:
    @staticmethod
    def plot_optuna_study(
        study: optuna.Study,
        viz_dir: Path,
    ) -> None:
        _ = optuna.visualization.plot_param_importances(study).write_image(
            viz_dir / "param_importance.png"
        )
        return None

    @staticmethod
    def plot_target_dist(data: pl.DataFrame, directory: Path) -> None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        churn_response = (
            data.select(pl.col("Churn"))
            .to_series()
            .value_counts(sort=True, parallel=True)
        )
        ax.bar(
            x=churn_response.select("Churn").to_numpy().squeeze(),
            height=churn_response.select("count").to_numpy().squeeze(),
            color=["#FDB0C0", "#4A0100"],
        )

        ax.set_title(
            "Proportion of observations of the response variable",
            fontsize=17,
            loc="center",
        )
        ax.set_xlabel("churn", fontsize=14)
        ax.set_ylabel("proportion of observations", fontsize=13)
        ax.tick_params(rotation="auto")
        plt.savefig(directory / "target_dist.png", format="png")
        plt.close()
        return None

    @staticmethod
    def plot_cust_info(data: pl.DataFrame, viz_dir: Path) -> None:
        colors = ["#E94B3C", "#2D2926"]
        l1 = ["gender", "SeniorCitizen", "Partner", "Dependents"]
        _ = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        for i in range(len(l1)):
            plt.subplot(2, 2, i + 1)
            ax = sns.countplot(
                x=l1[i],
                data=data.to_pandas(),
                hue="Churn",
                palette=colors,
                edgecolor="black",
            )
            for rect in ax.patches:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 2,
                    rect.get_height(),
                    horizontalalignment="center",
                    fontsize=11,
                )
            title = l1[i].capitalize() + " vs Churn"
            plt.title(title)
        plt.suptitle("Customer information")
        plt.savefig(viz_dir / "cust_info.png", format="png")
        plt.close()
        return None

    @staticmethod
    def plot_num_dist(data: pl.DataFrame, viz_dir: Path) -> None:
        num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        _, _ = plt.subplots(nrows=1, ncols=3, figsize=(11, 5))
        for i in range(len(num_features)):
            plt.subplot(1, 3, i + 1)
        sns.distplot(data[num_features[i]], color="#778da9")
        title = "Distribution : " + num_features[i]
        plt.title(title)
        plt.savefig(viz_dir / "num_cols_dist.png", format="png")
        plt.close()
        return None

    # FIXME
    @staticmethod
    def plot_training_results(config: Box, results: pl.DataFrame, viz_dir: Path):
        arr_labels = results.columns
        results: t.Dict[str, t.List[float]] = results.transpose().to_dict(
            as_series=False
        )
        models_lst = [
            list(model_item.keys())[0] for model_item in config.model.train.models
        ]
        values = list(results.values())
        num_bars = len(values[0])

        positions = np.arange(len(models_lst))
        group_width = 0.4
        width = 0.2
        plt.figure(figsize=(22, 7))
        for i, label in zip(range(num_bars), arr_labels):
            plt.bar(
                positions + i * group_width,
                [v[i] for v in values],
                width=width,
                label=label,
            )

        plt.xlabel("Models")
        plt.ylabel("Metrics")
        plt.title("Training results")
        plt.xticks(positions + width * (num_bars + 1) / 2, models_lst)
        plt.legend()
        plt.savefig(viz_dir / "training_results.png", format="png")
        plt.close()
