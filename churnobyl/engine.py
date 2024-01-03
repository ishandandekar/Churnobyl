"""
Pipeline workflow to retrain models on data.
This file has a whole range of functions that represent tasks in a MLOps retraining process
"""

import argparse
import random
import typing as t
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import wandb
import yaml
from box import Box
from prefect import artifacts, flow, get_run_logger, task

from data import DataEngine, TransformerOutput
from model import LearnLab, TunerOutput
from visualize import Vizard


@task(
    name="load_config",
    description="Function to load configuration settings from `.yaml` file",
)
def set_config(config_path: Path) -> Box:
    """
    Sets up configuration variables for pipeline using `.yaml` file

    Args:
        config_path (Path): Path for the `.yaml` file

    Raises:
        Exception: If the `.yaml` file is not present at `arg: config_path`

    Returns:
        Munch: object for better config variable calling
    """
    if config_path.exists():
        with open(config_path, "r") as stream:
            return Box(yaml.safe_load(stream=stream))
    else:
        raise Exception("Path error occured. File does not exist")


@task(
    name="setup_pipeline",
    description="Setup directories and logging for the pipeline experiment",
)
def setup_pipeline(
    config: Box,
) -> t.Tuple[Path, Path, Path, Path,]:
    """
    Creates directories and sets random seed for reproducibility

    Args:
        config (Munch): Configuraton variable mapping

    Returns:
        t.Tuple[Path, Path, Path, Path, Path, Path,]: Paths for all the directories
    """
    ROOT_DIR: Path = Path.cwd()
    VIZ_DIR: Path = ROOT_DIR / config.PATH.viz
    MODEL_DIR: Path = ROOT_DIR / config.PATH.model
    ARTIFACT_DIR: Path = ROOT_DIR / config.PATH.model / "artifacts"

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    return (
        ROOT_DIR,
        VIZ_DIR,
        MODEL_DIR,
        ARTIFACT_DIR,
    )


@task(
    name="data_loader",
    description="Load your data here and return one single dataframe",
    retries=3,
    retry_delay_seconds=3,
)
def data_loader(config: Box) -> pl.DataFrame:
    """
    Loads data from functions mentioned by the developer,
    Also validates data based on schema

    Args:
        config (Box): Configuration
        schema (pa.DataFrameSchema): Training data schema

    Raises:
        Exception: If the data does not match the schema

    Returns:
        pd.DataFrame: concatenated data that now be split into train and test sets
    """
    return DataEngine.load(config.data.load)


def data_validator(data) -> pl.DataFrame:
    return DataEngine.validate(data)


@task(
    name="data_split",
    description="Split data into training and test sets",
)
def data_splits(
    config: Box, data: pl.DataFrame
) -> t.Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits the data into train and test sets

    Args:
        config (Munch): configuration mapping
        df (pd.DataFrame): Data that is needed to be split

    Returns:
        Training features, test features, train labels, test labels
    """
    return DataEngine.split(config.data.split, data=data)


@task(
    name="data_transformer",
    description="Transform data into numerical encoding using preprocessors",
)
def data_transformer(
    config: Box,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    artifact_dir: Path,
) -> TransformerOutput:
    """
    Applies transformation like scaling and encoding to data
    """
    return DataEngine.transform(
        config=config.data.transform,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifact_dir=artifact_dir,
    )


@task(
    name="train_models",
    description="Trains model based on params, returns a dataframe containing metrics",
)
def train_models(config: Box, transformed_ds: TransformerOutput) -> pl.DataFrame:
    return LearnLab.train_experiments(
        config=config.model.train, transformed_ds=transformed_ds
    )


@task
def tune_models(
    config: Box, transformed_ds: t.Type[TransformerOutput], model_dir: Path
) -> TunerOutput:
    return LearnLab.tune_models(
        config=config.model.tune, transformed_ds=transformed_ds, model_dir=model_dir
    )


@task(
    name="make_plots_and_viz",
    description="Create visualizations for model explainability and data analysis",
)
def vizard(
    df: pl.DataFrame,
    results: pl.DataFrame,
    study: optuna.Study,
    viz_dir: Path,
) -> None:
    """
    Creates plots and visualizations for data analysis, results of training and hyper-parameter tuning

    Args:
        df (pd.DataFrame): Data for this pipeline
        results (pd.DataFrame): Training results
        study (optuna.Study): Results of hyper-parameter tuning
        viz_dir (Path): Directories to store all these visualizations
    """
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
    return None


@task(
    name="push_artifacts",
    description="Push artifacts to W&B server and Prefect server",
    retries=3,
    retry_delay_seconds=3,
)
def push_artifacts(
    best_model_name,
    best_metric: float,
    best_path_: Path,
    artifact_dir: Path,
    viz_dir: Path,
) -> None:
    """
    Pushes various artifacts such as log files, visualizations and models to respective servers and storage spaces
    """

    with wandb.init(project="churnobyl", job_type="pipeline") as run:
        model_artifact = wandb.Artifact("churnobyl-clf", type="model")
        model_artifact.add_file(str(best_path_))
        run.log_artifact(model_artifact)
        preprocessors_artifact = wandb.Artifact(
            "churnobyl-ohe-oe-stand", type="preprocessors"
        )
        preprocessors_artifact.add_dir(str(artifact_dir))
        run.log_artifact(preprocessors_artifact)
        plots_artifact = wandb.Artifact("plots", type="visualizations")
        plots_artifact.add_dir(str(viz_dir))
        run.log_artifact(plots_artifact)

    markdown_artifact = f"""
    ### Model saved: {best_model_name}
    ### Model performance: {best_metric}
    """
    _ = artifacts.create_markdown_artifact(
        key="model-report",
        markdown=markdown_artifact,
        description="Model summary report",
    )
    logger = get_run_logger()
    logger.info("Artifacts have been pushed to project server")
    logger.info("All tasks done. Pipeline has now been completed")
    return None


@flow(
    name="Churnobyl_retraining_pipeline_workflow",
    description="Pipeline for automated ml workflow",
)
def main_workflow(config_path: Path) -> None:
    """
    Entire pipeline, uses native Python logging

    Args:
        config_path (Path): Path for config file
    """
    config = set_config(config_path=config_path)
    (
        ROOT_DIR,
        VIZ_DIR,
        MODEL_DIR,
        ARTIFACT_DIR,
    ) = setup_pipeline(config=config)
    logger = get_run_logger()
    logger.info("Setting up directories and logging")
    df = data_loader(config=config)
    logger.info("Data has been loaded")
    X_train, X_test, y_train, y_test = data_splits(config=config, df=df)
    logger.info("Data splits have been made")
    transformed_ds = data_transformer(
        config=config,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifact_dir=ARTIFACT_DIR,
    )
    logger.info("Data transformers have been applied")

    results = train_models(config=config, transformed_ds=transformed_ds)
    tuned = tune_models(
        config=config, transformed_ds=transformed_ds, model_dir=MODEL_DIR
    )
    logger.info("Best model has been acquired")
    _ = vizard(
        df=df,
        results=results,
        study=tuned.studies[0],
        model=tuned.best_models[0],
        X_train=transformed_ds.X_train,
        viz_dir=VIZ_DIR,
    )
    logger.info("Visualizations have been drawn")
    _ = push_artifacts(
        best_type_=tuned.names[0],
        best_metric=tuned.best_metrics[0],
        best_path_=tuned.best_paths[0],
        artifact_dir=ARTIFACT_DIR,
        viz_dir=VIZ_DIR,
    )


if __name__ == "__main__":
    assert (
        Path.cwd().stem == "churninator"
    ), "Run code from 'churninator', not from `churnobyl`"
    parser = argparse.ArgumentParser(
        prog="Churnobyl-69420",
        description="For config file only",
    )
    parser.add_argument("--config", default="./churnobyl/conf/config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    main_workflow(config_path=config_path)
