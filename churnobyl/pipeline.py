import argparse
import logging
import pickle as pkl
import random
import typing as t
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import optuna
import pandas as pd
import pandera as pa
import wandb
import xgboost as xgb
import yaml
from model import LearnLab
from munch import Munch
from prefect import flow, task
from sklearn import (
    compose,
    dummy,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    neighbors,
    pipeline,
    preprocessing,
    svm,
    tree,
)
from visualize import Vizard

from data import TRAINING_SCHEMA, DataDreamer


@task(
    name="load_config",
    description="Function to load configuration settings from `.yaml` file",
)
def config(path: Path) -> Munch:
    if path.exists():
        with open(path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                return Munch(config)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        raise Exception("Path error occured. File does not exist")


@task(
    name="setup_pipeline",
    description="Setup directories and logging for the pipeline experiment",
)
def setup(
    config: Munch,
) -> t.Tuple[logging.Logger, Path, Path, Path, Path, Path, Path, pa.DataFrameSchema]:
    ROOT_DIR: Path = Path.cwd().parent

    date = datetime.today()
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(f"{config.paths.logs}/{date}.log").setLevel(
        logging.INFO
    )
    f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.info("Configuration loaded")

    DATA_DIR: Path = ROOT_DIR / config.paths.data
    VIZ_DIR: Path = ROOT_DIR / config.paths.viz
    MODEL_DIR: Path = ROOT_DIR / config.paths.model
    ARTIFACT_DIR: Path = ROOT_DIR / config.paths.model / "artifacts"
    LOGS_DIR: Path = ROOT_DIR / config.paths.logs

    logger.info("Setup completed")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    return (
        logger,
        ROOT_DIR,
        DATA_DIR.mkdir(parents=True, exist_ok=True),
        VIZ_DIR.mkdir(parents=True, exist_ok=True),
        MODEL_DIR.mkdir(parents=True, exist_ok=True),
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True),
        LOGS_DIR.mkdir(parents=True, exist_ok=True),
    )


@task(
    name="data_loader",
    description="Load your data here and return one single dataframe",
    retries=3,
    retry_delay_seconds=3,
)
def data_loader(schema: pa.DataFrameSchema, logger: logging.Logger) -> pd.DataFrame:
    columns = schema.columns.keys()
    # TODO: Add code to load data
    df = pd.DataFrame()
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)
        print("\nDataFrame object that failed validation:")
        print(err.data)
    logger.info("Data has been loaded")
    return df
    ...


@task(
    name="data_split",
    description="Split data into training and test sets",
)
def data_splits(config, df: pd.DataFrame, logger: logging.Logger):
    X, y = df.drop(columns=["Churn"]), df[["Churn"]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=config.data.train_test_split,
        random_state=config.SEED,
        stratify=y,
    )
    logger.info("Data has been split")
    return X_train, X_test, y_train, y_test


@task(
    name="data_transformer",
    description="Transform data into numerical encoding using preprocessors",
)
def data_transformer(
    config: Munch,
    X_train,
    X_test,
    y_train,
    y_test,
    artifact_dir: Path,
    logger: logging.Logger,
):
    # TODO: Train preprocessors
    # TODO: pickle trained preprocessors
    # TODO: Feed data for modelling from here
    preprocessor_names = []
    preprocessors = []
    for name, preprocessor in zip(preprocessor_names, preprocessors):
        path_ = artifact_dir / f"{name}.pkl"
        with open(str(path_), "wb") as f:
            pkl.dump(preprocessor, f)
    logger.info("Preprocessors have been saved into a `pickle` file")
    X_to_train, X_to_test = ..., ...
    logger.info("Data has been transformed")
    return X_to_train, X_to_test, y_train, y_test


@task(
    name="make_models",
    description="Fit multiple models on data and tune best models using optuna",
)
def get_best_model(
    config, X_train, X_test, y_train, y_test, model_dir: Path, logger: logging.Logger
) -> t.Tuple[
    pd.DataFrame,
    optuna.Study,
    t.Union[ensemble.RandomForestClassifier, xgb.XGBClassifier],
    t.Dict,
    float,
]:
    best_model, best_params, best_metric = None, None, None
    # TODO: Add code for best params and model
    # TODO: returns shap values too
    # TODO: Returns optuna study too, for vizualizations
    models = [
        (
            "dummy",
            dummy.DummyClassifier(random_state=config.SEED, strategy="most_frequent"),
        ),
        ("knn", neighbors.KNeighborsClassifier()),
        (
            "lr",
            linear_model.LogisticRegression(
                random_state=config.SEED, solver="liblinear", class_weight="balanced"
            ),
        ),
        ("svm", svm.SVC(random_state=config.SEED, kernel="rbf")),
        ("rf", ensemble.RandomForestClassifier(random_state=config.SEED)),
        (
            "gb",
            ensemble.GradientBoostingClassifier(random_state=config.SEED),
        ),
        ("dt", tree.DecisionTreeClassifier(random_state=config.SEED)),
        ("abc", ensemble.AdaBoostClassifier()),
        (
            "voting",
            ensemble.VotingClassifier(
                estimators=[
                    ("gbc", ensemble.GradientBoostingClassifier()),
                    ("lr", linear_model.LogisticRegression()),
                    ("abc", ensemble.AdaBoostClassifier()),
                ],
                voting="soft",
            ),
        ),
        ("xgb", xgb.XGBClassifier()),
    ]
    results: pd.DataFrame = LearnLab.run_experiments(
        model_list=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_dir=model_dir,
    )
    logger.info("Models have been trained")
    logger.info(
        f"{results.head(1).index.values[0].capitalize()} has been evaluated as the best model"
    )
    rf_study, rf_best_model, rf_best_params = ...
    xgb_study, xgb_best_model, xgb_best_params = ...
    # TODO: Comapre metrics of both the models
    study = ...
    logger.info("Best model acquired")
    return results, study, best_model, best_params, best_metric
    ...


@task(
    name="make_plots_and_viz",
    description="Create visualizations for model explainability and data analysis",
)
def vizard(
    df: pd.DataFrame,
    results: pd.DataFrame,
    study: optuna.Study,
    model,
    X_train: pd.DataFrame,
    viz_dir: Path,
    logger: logging.Logger,
) -> None:
    target_dist_path = viz_dir / "target_dist.png"
    contract_dist_path = viz_dir / "contract_dist.png"
    payment_dist_path = viz_dir / "payment_dist.png"
    isp_gender_churn_dist_path = viz_dir / "isp_gender_churn_dist.png"
    partner_churn_dist_path = viz_dir / "partner_churn_dist.png"
    Vizard.plot_data_insights(
        df=df,
        target_dist_path=target_dist_path,
        contract_dist_path=contract_dist_path,
        payment_dist_path=payment_dist_path,
        isp_gender_churn_dist_path=isp_gender_churn_dist_path,
        partner_churn_dist_path=partner_churn_dist_path,
    )
    performance_metrics_path = viz_dir / "performance_metrics.png"
    Vizard.plot_performance_metrics(results=results, path=performance_metrics_path)
    param_importance_path = viz_dir / "param_importance.png"
    parallel_coordinate_path = viz_dir / "parallel_coordinate.png"
    Vizard.plot_optuna(
        study=study,
        param_importance_path=param_importance_path,
        parallel_coordinate_path=parallel_coordinate_path,
    )
    shap_explainer_path = viz_dir / "shap_explainer.png"
    Vizard.plot_shap(model=model, X_train=X_train, path=shap_explainer_path)
    logger.info("Plots have been drawn")
    return None


@task(
    name="push_artifacts",
    description="Push artifacts to W&B server and Prefect server",
    retries=3,
    retry_delay_seconds=3,
)
def push_artifacts(logger: logging.Logger):
    # TODO: Push artifacts to WandB server
    logger.info("Artifacts have been pushed to project server")
    ...


@flow(
    name="Churnobyl_retraining_pipeline_workflow",
    description="Pipeline for automated ml workflow",
)
def main(config_path: Path) -> None:
    config = config(config_path)
    (
        logger,
        ROOT_DIR,
        DATA_DIR,
        VIZ_DIR,
        MODEL_DIR,
        ARTIFACT_DIR,
        LOGS_DIR,
    ) = setup(config=config)
    print("[INFO] Setup completed")
    df = data_loader(schema=TRAINING_SCHEMA, logger=logger)
    X_train, X_test, y_train, y_test = data_splits(config=config, df=df, logger=logger)
    X_to_train, X_to_test, y_train, y_test = data_transformer(
        config=config,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifact_dir=ARTIFACT_DIR,
        logger=logger,
    )
    results, study, best_model, best_params, best_metric = get_best_model(
        config=config,
        X_train=X_to_train,
        X_test=X_to_test,
        y_train=y_train,
        y_test=y_test,
        model_dir=MODEL_DIR,
    )
    logger.info("All processes done. Pipeline has been completed")
    ...


if __name__ == "__main__":
    assert (
        Path.cwd().stem == "churninator"
    ), "Run code from parent directory, not from 'churnobyl'"
    parser = argparse.ArgumentParser(
        prog="PregnantHippo-69420",
        description="For config file only",
    )
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    main(config_path=config)
