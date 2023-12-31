"""
Pipeline workflow to retrain models on data.
This file has a whole range of functions that represent tasks in a MLOps retraining process
"""

import argparse
import random
import typing as t
from pathlib import Path
import boto3
import cloudpickle as cpickle
import numpy as np
import optuna
import pandas as pd
import pandera as pa
import wandb
import xgboost as xgb
import yaml
from box import Box
from prefect import flow, task, artifacts, get_run_logger
from sklearn import (
    ensemble,
    model_selection,
    preprocessing,
)
import numpy.typing as npt
from data import TRAINING_SCHEMA, DataLoaderStrategyFactory
from visualize import Vizard
from model import LearnLab, ModelFactory


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
    LOGS_DIR: Path = ROOT_DIR / config.PATH.logs
    VIZ_DIR: Path = ROOT_DIR / config.PATH.viz
    MODEL_DIR: Path = ROOT_DIR / config.PATH.model
    ARTIFACT_DIR: Path = ROOT_DIR / config.PATH.model / "artifacts"

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    return (
        ROOT_DIR,
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
def get_best_model(
    config, X_train, X_test, y_train, y_test, model_dir: Path
) -> t.Tuple[
    pd.DataFrame,
    optuna.Study,
    t.Union[ensemble.RandomForestClassifier, xgb.XGBClassifier],
    t.Dict,
    float,
]:
    """
    Trains multiple models as mentioned in `arg: config` and tunes random forest and XGBoost model

    Args:
        config (_type_): Configuration mapping
        X_train (_type_): Training features
        X_test (_type_): Test features
        y_train (_type_): Training labels
        y_test (_type_): Test labels
        model_dir (Path): Directories where the tuned models should be stored

    Returns:
        t.Tuple[ pd.DataFrame, optuna.Study, t.Union[ensemble.RandomForestClassifier, xgb.XGBClassifier], t.Dict, float]: Gives results of training, tuning experiment, best model, best model's hyperparameters, metric performance of the best model, path of the best model, type of the model i.e. random forest or XGBoost
    """
    models = list()
    for model_name in config.model.get("models"):
        if model_name == "voting":
            name, voting_model = ModelFactory.get(model_name)
            estimators = list()
            for voting_model_name in config.model.get("models").get("voting"):
                estimators.append(ModelFactory.get(voting_model_name))
            params = {"estimators": estimators, "voting": "soft"}
            voting_model = voting_model.set_params(**params)
            models.append((name, voting_model))
        models.append(ModelFactory.get(model_name))
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
        cpickle.dump(best_model, f)
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
    """
    Creates plots and visualizations for data analysis, results of training and hyper-parameter tuning

    Args:
        df (pd.DataFrame): Data for this pipeline
        results (pd.DataFrame): Training results
        study (optuna.Study): Results of hyper-parameter tuning
        model (_type_): Either Random Forest or XGBoost
        X_train (pd.DataFrame): Training features
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
    # shap_explainer_path = viz_dir / "shap_explainer.png"
    # Vizard.plot_shap(model=model, X_train=X_train, path=shap_explainer_path)
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
    # logger: logging.Logger,
    # logger_file_handler,
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
    ### Model saved: {best_type_}
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
    # logger_file_handler.close()
    # s3_resource = boto3.resource("s3")
    # bucket = s3_resource.Bucket("churnobyl")
    # log_files = logs_dir.glob("*.log")
    # for file in log_files:
    #     bucket.upload_file(file, f"train_logs/{file.name}")
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
        LOGS_DIR,
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
        viz_dir=VIZ_DIR,
        logs_dir=LOGS_DIR,
        # logger=logger,
        # logger_file_handler=file_handler,
    )


if __name__ == "__main__":
    assert (
        Path.cwd().stem == "churninator"
    ), "Run code from 'churninator', not from `churnobyl`"
    parser = argparse.ArgumentParser(
        prog="Churnzilla-69420",
        description="For config file only",
    )
    parser.add_argument("--config", default="./churnobyl/conf/config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    main_workflow(config_path=config_path)
