from pathlib import Path
from datetime import datetime
import logging
import typing as t
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import (
    model_selection,
    preprocessing,
    dummy,
    metrics,
    ensemble,
    tree,
    neighbors,
    pipeline,
    compose,
    linear_model,
    svm,
)
import optuna as opt
import shap
import pickle
import boto3
from prefect import task, flow
import argparse
import yaml
from munch import Munch
import wandb
from churnobyl import data, model


@task
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


@task
def setup(config_path: Path):
    ROOT_DIR = Path.cwd().parent.absolute()

    config = config(config_path)

    DATA_DIR: Path = ROOT_DIR / config.paths.data
    VIZ_DIR: Path = ROOT_DIR / config.paths.viz
    MODEL_DIR: Path = ROOT_DIR / config.paths.model
    LOGS_DIR: Path = ROOT_DIR / config.paths.logs

    date = datetime.today()
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(f"{config.paths.logs}/{date}.log").setLevel(
        logging.INFO
    )
    f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return (
        config,
        logger,
        ROOT_DIR,
        DATA_DIR.mkdir(parents=True, exist_ok=True),
        VIZ_DIR.mkdir(parents=True, exist_ok=True),
        MODEL_DIR.mkdir(parents=True, exist_ok=True),
        LOGS_DIR.mkdir(parents=True, exist_ok=True),
    )


@task
def data_loader():
    # TODO: Add transformations in `data.py`
    # TODO: Returns train test splits
    # TODO: Add logger
    ...


@task
def data_transformer():
    # TODO: Use preprocessor here itself
    # TODO: Load preprocessor here
    # TODO: Feed data for modelling from here
    ...


@task
def get_best_model(config, X_train, X_test, y_train, y_test):
    # TODO: Add code for best params and model
    # TODO: returns shap values too
    # TODO: Returns optuna study too, for vizualizations
    models = [
        (
            "dummy_classifier",
            dummy.DummyClassifier(random_state=config.SEED, strategy="most_frequent"),
        ),
        ("k_nearest_neighbors", neighbors.KNeighborsClassifier()),
        (
            "logistic_regression",
            linear_model.LogisticRegression(
                random_state=config.SEED, solver="liblinear", class_weight="balanced"
            ),
        ),
        ("support_vector_machines", svm.SVC(random_state=config.SEED, kernel="rbf")),
        ("random_forest", ensemble.RandomForestClassifier(random_state=config.SEED)),
        (
            "gradient_boosting",
            ensemble.GradientBoostingClassifier(random_state=config.SEED),
        ),
        ("decision_tree", tree.DecisionTreeClassifier(random_state=config.SEED)),
        ("adaboost", ensemble.AdaBoostClassifier()),
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
    ]
    results = []
    ...


@task
def vizard():
    # TODO: Accepts viz dir, model and shap
    # TODO: Makes vizualizations for optuna and shap
    ...


@task
def push_artifacts():
    # TODO: Push artifacts to WandB server
    ...


@flow
def main(config_path: Path) -> None:
    config, logger, ROOT_DIR, DATA_DIR, VIZ_DIR, MODEL_DIR, LOGS_DIR = setup(
        config_path=config_path
    )
    print("[INFO] Setup completed")
    logger.info("Setup completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PregnantHippo-69420",
        description="For config file only",
    )
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    main(config_path=config)
