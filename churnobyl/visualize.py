"""
Contains functions to make plots and visualizations
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import figure
import optuna
import pandas as pd
import shap
from plotly import express as px
from plotly import graph_objs as go


class Vizard:
    # DEV: Add plots for data analysis here
    @staticmethod
    def plot_data_insights(
        df: pd.DataFrame,
        target_dist_path: Path,
        contract_dist_path: Path,
        payment_dist_path: Path,
        isp_gender_churn_dist_path: Path,
        partner_churn_dist_path: Path,
    ) -> None:
        for path in [
            target_dist_path,
            contract_dist_path,
            payment_dist_path,
            isp_gender_churn_dist_path,
            partner_churn_dist_path,
        ]:
            if not str(path).endswith(".png"):
                raise Exception(f"{path} should be a path to `.png`")

        # Target distribution
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        churn_response = df["Churn"].value_counts(normalize=True)
        churn_response.plot(kind="bar", ax=ax, color=["#101820", "#FEE715"])
        ax.set_title(
            "Proportion of observations of the response variable",
            fontsize=17,
            loc="left",
        )
        ax.set_xlabel("churn", fontsize=14)
        ax.set_ylabel("proportion of observations", fontsize=13)
        ax.tick_params(rotation="auto")
        plt.savefig(target_dist_path, format="png")
        plt.close()

        fig = px.histogram(
            df,
            x="Churn",
            color="Contract",
            barmode="group",
            title="<b>Customer contract distribution<b>",
        )
        fig.update_layout(width=700, height=500, bargap=0.2)
        fig.write_image(contract_dist_path)

        labels = df["PaymentMethod"].unique()
        values = df["PaymentMethod"].value_counts()

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        fig.update_layout(title_text="<b>Payment Method Distribution</b>")
        fig.write_image(payment_dist_path)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[
                    ["Churn:No", "Churn:No", "Churn:Yes", "Churn:Yes"],
                    ["Female", "Male", "Female", "Male"],
                ],
                y=[965, 992, 219, 240],
                name="DSL",
            )
        )

        fig.add_trace(
            go.Bar(
                x=[
                    ["Churn:No", "Churn:No", "Churn:Yes", "Churn:Yes"],
                    ["Female", "Male", "Female", "Male"],
                ],
                y=[889, 910, 664, 633],
                name="Fiber optic",
            )
        )

        fig.add_trace(
            go.Bar(
                x=[
                    ["Churn:No", "Churn:No", "Churn:Yes", "Churn:Yes"],
                    ["Female", "Male", "Female", "Male"],
                ],
                y=[690, 717, 56, 57],
                name="No Internet",
            )
        )

        fig.update_layout(
            title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>"
        )

        fig.write_image(isp_gender_churn_dist_path)

        color_map = {"Yes": "#FFA15A", "No": "#00CC96"}
        fig = px.histogram(
            df,
            x="Churn",
            color="Partner",
            barmode="group",
            title="<b>Chrun distribution w.r.t. Partners</b>",
            color_discrete_map=color_map,
        )
        fig.update_layout(width=700, height=500, bargap=0.1)
        fig.write_image(partner_churn_dist_path)
        return None

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
        return None

    # DEV: Add `optuna.visualizations` as per need
    @staticmethod
    def plot_optuna(
        study: optuna.Study, param_importance_path: Path, parallel_coordinate_path: Path
    ) -> None:
        if not str(param_importance_path).endswith(".png"):
            raise Exception(f"{param_importance_path} should be a path to `.png`")
        if not str(parallel_coordinate_path).endswith(".png"):
            raise Exception(f"{parallel_coordinate_path} should be a path to `.png`")
        param_importances: go.Figure = optuna.visualization.plot_param_importances(
            study
        )
        param_importances.write_image(param_importance_path)

        parallel_coordinate: go.Figure = optuna.visualization.plot_parallel_coordinate(
            study
        )
        parallel_coordinate.write_image(parallel_coordinate_path)
        return None

    # DEV: Add shap plots as per need here
    @staticmethod
    def plot_shap(model, X_train: pd.DataFrame, path: Path) -> None:
        if not str(path).endswith(".png"):
            raise Exception(f"{path} should be a path to `.png`")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        shap.plots.bar(shap_values, show=False)
        fig = plt.gcf()
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return None
