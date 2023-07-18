"""
Pipeline workflow to retrain models on data.
This file has a whole range of functions that represent tasks in a MLOps retraining process
"""

import argparse
import logging
import pickle as pkl
import random
import typing as t
import os
from datetime import datetime
from pathlib import Path
from glob import glob
import boto3
import numpy as np
import optuna
import pandas as pd
import pandera as pa
import wandb
import xgboost as xgb
import yaml
from munch import Munch
from prefect import flow, task, artifacts
from sklearn import (
    ensemble,
    model_selection,
    preprocessing,
)

from data import TRAINING_SCHEMA, DataDreamer
from visualize import Vizard
from model import LearnLab, MODEL_DICT


def _custom_combiner(feature, category):
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)


@task(
    name="load_config",
    description="Function to load configuration settings from `.yaml` file",
)
def set_config(config_path: Path) -> Munch:
    if config_path.exists():
        with open(config_path, "r") as stream:
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
def setup_pipeline(
    config: Munch,
) -> t.Tuple[Path, Path, Path, Path, Path, Path,]:
    ROOT_DIR: Path = Path.cwd()
    LOGS_DIR: Path = ROOT_DIR / config.PATH.get("logs")

    DATA_DIR: Path = ROOT_DIR / config.PATH.get("data")
    VIZ_DIR: Path = ROOT_DIR / config.PATH.get("viz")
    MODEL_DIR: Path = ROOT_DIR / config.PATH.get("model")
    ARTIFACT_DIR: Path = ROOT_DIR / config.PATH.get("model") / "artifacts"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    return (
        ROOT_DIR,
        DATA_DIR,
        VIZ_DIR,
        MODEL_DIR,
        ARTIFACT_DIR,
        LOGS_DIR,
    )


@task(
    name="data_loader",
    description="Load your data here and return one single dataframe",
    retries=3,
    retry_delay_seconds=3,
)
def data_loader(schema: pa.DataFrameSchema, data_dir: t.Optional[Path]) -> pd.DataFrame:
    columns = schema.columns.keys()
    df = DataDreamer.load_csv_from_dir(dir=data_dir, columns=columns)
    df.TotalCharges = df.TotalCharges.replace(to_replace=" ", value="0")
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)
        print("\nDataFrame object that failed validation:")
        print(err.data)
        raise Exception("Schema errors and failure cases")
    df.TotalCharges = df.TotalCharges.replace(to_replace=" ", value="0")

    return df


@task(
    name="data_split",
    description="Split data into training and test sets",
)
def data_splits(config, df: pd.DataFrame):
    X, y = df.drop(columns=["Churn"]), df[["Churn"]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=config.data.get("train_test_split"),
        random_state=config.SEED,
        stratify=y,
    )

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
):
    scaler = preprocessing.StandardScaler().set_output(transform="pandas")
    encoder_ohe = preprocessing.OneHotEncoder(feature_name_combiner=_custom_combiner)
    encoder_oe = preprocessing.OrdinalEncoder().set_output(transform="pandas")
    target_encoder = preprocessing.LabelEncoder()

    # One-hot encoding
    X_ohe__train = X_train[config.data.get("CAT_COLS_OHE")]
    encoder_ohe.fit(X_ohe__train)
    X_ohe_trans__train = encoder_ohe.transform(X_ohe__train)
    X_ohe_trans_df__train: pd.DataFrame = pd.DataFrame(
        X_ohe_trans__train.toarray(), columns=encoder_ohe.get_feature_names_out()
    )
    X_ohe__test = X_test[config.data.get("CAT_COLS_OHE")]
    X_ohe_trans__test = encoder_ohe.transform(X_ohe__test)
    X_ohe_trans__test: pd.DataFrame = pd.DataFrame(
        X_ohe_trans__test.toarray(), columns=encoder_ohe.get_feature_names_out()
    )

    # Ordinal Encoder
    X_oe__train = X_train[config.data.get("CAT_COLS_OE")]
    encoder_oe.fit(X_oe__train)
    X_oe_trans__train: pd.DataFrame = encoder_oe.transform(X_oe__train)
    X_oe__test = X_test[config.data.get("CAT_COLS_OE")]
    X_oe_trans__test: pd.DataFrame = encoder_oe.transform(X_oe__test)

    # Scale
    X_scale__train = X_train[config.data.get("NUM_COLS")]
    scaler.fit(X_scale__train)
    X_scale_trans__train: pd.DataFrame = scaler.transform(X_scale__train)
    X_scale__test = X_test[config.data.get("NUM_COLS")]
    X_scale_trans__test: pd.DataFrame = scaler.transform(X_scale__test)

    # Encode target variable
    target_encoder.fit(y_train)
    y_to_train = target_encoder.transform(y_train)
    y_to_test = target_encoder.transform(y_test)

    X_to_train: pd.DataFrame = pd.concat(
        [
            X_ohe_trans_df__train.reset_index(drop=True),
            X_oe_trans__train.reset_index(drop=True),
            X_scale_trans__train.reset_index(drop=True),
        ],
        axis=1,
    )
    X_to_test: pd.DataFrame = pd.concat(
        [
            X_ohe_trans__test.reset_index(drop=True),
            X_oe_trans__test.reset_index(drop=True),
            X_scale_trans__test.reset_index(drop=True),
        ],
        axis=1,
    )

    preprocessors_names = [
        "encoder_ohe_",
        "encoder_oe_",
        "scaler_standard_",
        "target_encoder_",
    ]
    preprocessors = [encoder_ohe, encoder_oe, scaler, target_encoder]
    for name, preprocessor in zip(preprocessors_names, preprocessors):
        path_ = artifact_dir / f"{name}.pkl"
        with open(str(path_), "wb") as f:
            pkl.dump(preprocessor, f, pkl.HIGHEST_PROTOCOL)
    return X_to_train, X_to_test, y_to_train, y_to_test


@task(
    name="make_models",
    description="Fit multiple models on data and tune best models using optuna",
    retries=1,
)
def get_best_model(
    config, X_train, X_test, y_train, y_test, model_dir: Path
) -> t.Tuple[
    pd.DataFrame,
    optuna.Study,
    t.Union[ensemble.RandomForestClassifier, xgb.XGBClassifier],
    t.Dict,
    float,
]:
    models = list()
    for model_name in config.model.get("models"):
        if model_name == "voting":
            name, voting_model = MODEL_DICT.get(model_name)
            estimators = list()
            for voting_model_name in config.model.get("models").get("voting"):
                estimators.append(MODEL_DICT.get(voting_model_name))
            params = {"estimators": estimators, "voting": "soft"}
            voting_model = voting_model.set_params(**params)
            models.append((name, voting_model))
        models.append(MODEL_DICT.get(model_name))
    results: pd.DataFrame = LearnLab.run_experiments(
        model_list=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    study, best_model, best_params, best_metric, type_ = LearnLab.tune_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_trials=config.model.get("n_trials"),
    )
    best_path_ = model_dir / f"{type_}_best_.pkl"
    with open(best_path_, "wb") as f:
        pkl.dump(best_model, f)
    return results, study, best_model, best_params, best_metric, best_path_, type_


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
    return None


@task(
    name="push_artifacts",
    description="Push artifacts to W&B server and Prefect server",
    retries=3,
    retry_delay_seconds=3,
)
def push_artifacts(
    best_type_,
    best_metric: float,
    best_path_: Path,
    artifact_dir: Path,
    viz_dir: Path,
    logs_dir: Path,
    logger: logging.Logger,
    logger_file_handler,
):
    wandb.init(project="churnobyl", job_type="pipeline")
    model_artifact = wandb.Artifact("churnobyl-clf", type="model")
    model_artifact.add_file(best_path_)
    wandb.log_artifact(model_artifact)
    preprocessors_artifact = wandb.Artifact(
        "churnobyl-ohe-oe-stand", type="preprocessors"
    )
    preprocessors_artifact.add_dir(artifact_dir)
    wandb.log_artifact(preprocessors_artifact)
    plots_artifact = wandb.Artifact("plots", type="visualizations")
    plots_artifact.add_dir(viz_dir)
    wandb.log_artifact(plots_artifact)
    markdown_artifact = f"""
    ### Model saved: {best_type_}
    ### Model performance: {best_metric}
    """
    artifacts.create_markdown_artifact(
        key="model-report",
        markdown=markdown_artifact,
        description="Model summary report",
    )
    wandb.finish()
    logger.info("Artifacts have been pushed to project server")
    logger.info("All tasks done. Pipeline has now been completed")
    logger_file_handler.close()
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket("churnobyl")
    log_files = logs_dir.glob("*.log")
    for file in log_files:
        bucket.upload_file(file, f"train_logs/{file.name}")
    return None


@flow(
    name="Churnobyl_retraining_pipeline_workflow",
    description="Pipeline for automated ml workflow",
)
def main_workflow(config_path: Path) -> None:
    config = set_config(config_path=config_path)
    (
        ROOT_DIR,
        DATA_DIR,
        VIZ_DIR,
        MODEL_DIR,
        ARTIFACT_DIR,
        LOGS_DIR,
    ) = setup_pipeline(config=config)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(
        filename=LOGS_DIR / f"{datetime.now().date()}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Setting up directories and logging")
    df = data_loader(schema=TRAINING_SCHEMA, data_dir=DATA_DIR)
    logger.info("Data has been loaded")
    X_train, X_test, y_train, y_test = data_splits(config=config, df=df)
    logger.info("Data splits have been made")
    X_to_train, X_to_test, y_to_train, y_to_test = data_transformer(
        config=config,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifact_dir=ARTIFACT_DIR,
    )
    logger.info("Data transformers have been applied")
    (
        results,
        study,
        best_model,
        best_params,
        best_metric,
        best_path_,
        best_type_,
    ) = get_best_model(
        config=config,
        X_train=X_to_train,
        X_test=X_to_test,
        y_train=y_to_train,
        y_test=y_to_test,
        model_dir=MODEL_DIR,
    )
    logger.info("Best model has been acquired")
    _ = vizard(
        df=df,
        results=results,
        study=study,
        model=best_model,
        X_train=X_to_train,
        viz_dir=VIZ_DIR,
    )
    logger.info("Visualizations have been drawn")
    _ = push_artifacts(
        best_type_=best_type_,
        best_metric=best_metric,
        best_path_=best_path_,
        artifact_dir=ARTIFACT_DIR,
        logs_dir=LOGS_DIR,
        logger=logger,
        logger_file_handler=file_handler,
    )


if __name__ == "__main__":
    assert os.getenv(
        "WANDB_API_KEY"
    ), "You must set the WANDB_API_KEY environment variable"
    assert (
        Path.cwd().stem == "churninator"
    ), "Run code from 'churninator', not from 'churnobyl'"
    parser = argparse.ArgumentParser(
        prog="Churnzilla-69420",
        description="For config file only",
    )
    parser.add_argument("--config", default="./churnobyl/conf/config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    main_workflow(config_path=config_path)
