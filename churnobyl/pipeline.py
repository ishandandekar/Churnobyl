from pathlib import Path
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
from churnobyl import data as Data, model as Model


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
def data():
    ...


@task
def get_best_model():
    ...


@task
def vizard():
    ...


@task
def push_artifacts():
    ...


@flow
def main(config_path: Path) -> None:
    config = config(config_path)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Churnobyl-CheesBalls-69420",
        description="For config file only",
    )
    parser.add_argument("--config", default="./config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config_dir)
    main(config_path=config_path)
