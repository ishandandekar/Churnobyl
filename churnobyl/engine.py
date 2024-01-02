"""
Pipeline workflow to retrain models on data.
This file has a whole range of functions that represent tasks in a MLOps retraining process
"""

import argparse
import random
import typing as t
from pathlib import Path
import cloudpickle as cpickle
import numpy as np
import optuna
import pandas as pd
import pandera as pa
import wandb
import yaml
from box import Box
from prefect import flow, task, artifacts, get_run_logger
from sklearn import (
    model_selection,
    preprocessing,
)
import numpy.typing as npt
from data import TRAINING_SCHEMA, DataLoaderStrategyFactory
from visualize import Vizard
from model import LearnLab, ModelEngineOutput


def _custom_combiner(feature, category):
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)


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
            try:
                config = yaml.safe_load(stream)
                return Box(config)
            except yaml.YAMLError as exc:
                print(exc)
                return None
    else:
        raise Exception("Path error occured. File does not exist")


@task(
    name="setup_pipeline",
    description="Setup directories and logging for the pipeline experiment",
)
def setup_pipeline(
    config: Box,
) -> t.Tuple[Path, Path, Path, Path, Path, Path,]:
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
def data_loader(config: Box, schema: pa.DataFrameSchema) -> pd.DataFrame:
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
    data: pd.DataFrame = DataLoaderStrategyFactory.get(config.data.load.strategy)(
        **config.data.load.args
    )()
    data["TotalCharges"] = data["TotalCharges"].replace(to_replace=" ", value="0")
    try:
        schema.validate(data, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)
        print("\nDataFrame object that failed validation:")
        print(err.data)
        raise Exception("Schema errors and failure cases")
    return data


@task(
    name="data_split",
    description="Split data into training and test sets",
)
def data_splits(
    config: Box, df: pd.DataFrame
) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into train and test sets

    Args:
        config (Munch): configuration mapping
        df (pd.DataFrame): Data that is needed to be split

    Returns:
        Training features, test features, train labels, test labels
    """
    X, y = df.drop(columns=["Churn"]), df[["Churn"]]
    if config.data.split.stratify:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X,
            y,
            test_size=config.data.split.ratio,
            random_state=config.SEED,
            stratify=y,
        )
    else:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X,
            y,
            test_size=config.data.split.ratio,
            random_state=config.SEED,
        )

    return X_train, X_test, y_train, y_test


@task(
    name="data_transformer",
    description="Transform data into numerical encoding using preprocessors",
)
def data_transformer(
    config: Box,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    artifact_dir: Path,
) -> t.Tuple[pd.DataFrame, pd.DataFrame, npt.ArrayLike, npt.ArrayLike]:
    """
    Applies transformation like scaling and encoding to data
    """
    scaler = preprocessing.StandardScaler().set_output(transform="pandas")
    encoder_ohe = preprocessing.OneHotEncoder(feature_name_combiner=_custom_combiner)
    encoder_oe = preprocessing.OrdinalEncoder()
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
    encoder_oe.fit(X_oe__train.values)

    X_oe_trans__train = encoder_oe.transform(X_oe__train.values)
    X_oe_trans__train = pd.DataFrame(
        X_oe_trans__train, columns=config.data.get("CAT_COLS_OE")
    )
    X_oe__test = X_test[config.data.get("CAT_COLS_OE")]
    X_oe_trans__test = encoder_oe.transform(X_oe__test.values)
    X_oe_trans__test: pd.DataFrame = pd.DataFrame(
        X_oe_trans__test, columns=config.data.get("CAT_COLS_OE")
    )

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

    # Saving preprocessors as `.pkl` files
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
            cpickle.dumps(preprocessor, f)

    return X_to_train, X_to_test, y_to_train, y_to_test


@task(
    name="make_models",
    description="Fit multiple models on data and tune best models using optuna",
    retries=1,
)
def model_engine(
    config: Box,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: npt.ArrayLike,
    y_test: npt.ArrayLike,
    model_dir: Path,
) -> ModelEngineOutput:
    """
    Trains and tunes models based on `args: config`

    Args:
        config (Box): Configuration mapping for training and tuning
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (npt.ArrayLike): Training labels
        y_test (npt.ArrayLike): Test labels
        model_dir (Path): Directory to store tuned models

    Returns:
        ModelEngineOutput: Output class for training results, tuner output and paths for the tuned models
    """
    results: pd.DataFrame = LearnLab.train_experiments(
        config=config.model.train,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    tuner = LearnLab.tune_models(
        config=config.model.tune,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    best_paths = list()
    for model, name in zip(tuner.best_models, tuner.names):
        path_ = model_dir / f"{name}_tuned_.pkl"
        with open(path_, "wb") as f_out:
            cpickle.dump(model, f_out)
        best_paths.append(path_)
    return ModelEngineOutput(results=results, tuner=tuner, paths=best_paths)


@task(
    name="make_plots_and_viz",
    description="Create visualizations for model explainability and data analysis",
)
def vizard(
    df: pd.DataFrame,
    results: pd.DataFrame,
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
        model_artifact.add_file(best_path_)
        run.log_artifact(model_artifact)
        preprocessors_artifact = wandb.Artifact(
            "churnobyl-ohe-oe-stand", type="preprocessors"
        )
        preprocessors_artifact.add_dir(artifact_dir)
        run.log_artifact(preprocessors_artifact)
        plots_artifact = wandb.Artifact("plots", type="visualizations")
        plots_artifact.add_dir(viz_dir)
        run.log_artifact(plots_artifact)

    markdown_artifact = f"""
    ### Model saved: {best_model_name}
    ### Model performance: {best_metric}
    """
    artifacts.create_markdown_artifact(
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
    df = data_loader(config=config, schema=TRAINING_SCHEMA)
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
    model_engined_output = model_engine(
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
        results=model_engined_output.results,
        study=model_engined_output.tuner.studies[0],
        model=model_engined_output.tuner.best_models[0],
        X_train=X_to_train,
        viz_dir=VIZ_DIR,
    )
    logger.info("Visualizations have been drawn")
    _ = push_artifacts(
        best_type_=model_engined_output.tuner.names[0],
        best_metric=model_engined_output.tuner.best_metrics[0],
        best_path_=model_engined_output.paths[0],
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
