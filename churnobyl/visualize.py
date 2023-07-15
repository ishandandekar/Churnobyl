from pathlib import Path
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import shap
import math


class Vizard:
    @staticmethod
    def plot_data_insights(
        df: pd.DataFrame,
        target_dist_path: Path,
        demographic_dist_path: Path,
        cust_acc_dist_path: Path,
    ) -> None:
        for path in [target_dist_path, demographic_dist_path, cust_acc_dist_path]:
            if not str(path).endswith(".png"):
                raise Exception(f"{path} should be a path to `.png`")

        def percentage_plot(cols: list, sup_title: str, save_path: Path):
            n_cols = 2
            n_rows = math.ceil(len(cols) / 2)

            fig = plt.figure(figsize=(12, 5 * n_rows))
            fig.suptitle(sup_title, fontsize=22, y=0.95)

            for idx, col in enumerate(cols, start=1):
                ax = fig.add_subplot(n_rows, n_cols, idx)

                prop_by_independant = pd.crosstab(df[col], df["Churn"]).apply(
                    lambda x: x / x.sum() * 100, axis=1
                )
                prop_by_independant.plot(
                    kind="bar", ax=ax, stacked=True, rot=0, color=["#101820", "#FEE715"]
                )

                ax.legend(
                    loc="upper right",
                    bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                    title="Churn",
                    fancybox=True,
                )
                ax.set_title(
                    f"Proportion of obsevations by {col}", fontsize=16, loc="left"
                )
                ax.tick_params(rotation="auto")
                spine_names = ["top", "right"]
                for spine in spine_names:
                    ax.spines[spine].set_visible(False)
                plt.savefig(save_path, format="png")
                plt.close()

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
        spine_names = ["top", "right"]
        for spine in spine_names:
            ax.spines[spine].set_visible(False)
        plt.savefig(target_dist_path, format="png")
        plt.close()

        demographic_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
        percentage_plot(
            cols=demographic_cols,
            sup_title="Demographic Information",
            save_path=demographic_dist_path,
        )

        account_cols = ["Contract", "PaperlessBilling", "PaymentMethod"]
        percentage_plot(
            account_cols, "Customer Account Information", save_path=cust_acc_dist_path
        )

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
        param_importances = optuna.visualization.plot_param_importances(study)
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
