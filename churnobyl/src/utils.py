"""
This script includes all the helper utility functions and classes
"""
import os
import pickle
import random
import typing as t
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import yaml
from box import Box
from PIL import Image
from prefect import artifacts
from src.data import validate_data_config
from src.exceptions import ConfigValidationError
from src.model import TunerOutput, validate_model_config


class Pilot:
    @staticmethod
    def __validate_config(config: Box):
        """
        Validates configuration mapping

        Args:
            config (Box): Configuration mapping

        Raises:
            ConfigValidationError: If seed specified is not an integer
        """
        if not isinstance(config.get("SEED"), int):
            raise ConfigValidationError("SEED should be an integer.")
        validate_data_config(config=config)
        validate_model_config(config=config)

    @staticmethod
    def setup(filepath: t.Union[str, Path]) -> t.Tuple[Box, Path, Path, Path, Path]:
        """
        _summary_

        Args:
            filepath (t.Union[str, Path]): Path to the `.yaml` file

        Returns:
            t.Tuple[Box, Path, Path, Path, Path]: Configuration mapping and paths to various directories
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if (
            filepath.exists()
            and filepath.is_file()
            and filepath.suffix in [".yaml", ".yml"]
        ):
            with open(filepath, "r") as f_in:
                config = Box(yaml.safe_load(stream=f_in))

            Pilot.__validate_config(config=config)

            ROOT_DIR = Path.cwd().absolute()
            VIZ_DIR: Path = ROOT_DIR / config.path.viz
            MODEL_DIR: Path = ROOT_DIR / config.path.model
            ARTIFACT_DIR: Path = ROOT_DIR / config.path.model / "artifacts"

            VIZ_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

            random.seed(config.SEED)
            np.random.seed(config.SEED)
            return (
                config,
                ROOT_DIR,
                VIZ_DIR,
                MODEL_DIR,
                ARTIFACT_DIR,
            )
        else:
            FileNotFoundError(
                f"Please check if {filepath} is either `.yaml` or `.yml` is present on the correct location."
            )

    @staticmethod
    def populate_env_vars(filepath: t.Union[str, Path]) -> None:
        """
        Adds environment variables to runtime

        Args:
            filepath (t.Union[str, Path]): Path to the .env file
        """
        with open(filepath, "rb") as f_in:
            vars = f_in.readlines()

        for var in vars:
            k, v = var.decode().strip("\n").strip(" ").split("=")
            os.environ[k] = v
        return None

    @staticmethod
    def push_artifacts(tuner: TunerOutput, viz_dir: Path) -> None:
        # Saving tuning results to prefect artifacts
        tuner_table = tuner.as_table()
        with pl.Config(fmt_str_lengths=50):
            artifacts.create_table_artifact(
                key="tuning-results",
                table=tuner_table.to_dict(as_series=False),
                description="## Tuning results of the run",
            )
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(experiment_name="churnobyl")

        with mlflow.start_run():
            for image_path in viz_dir.glob("*.png"):
                mlflow.log_image(image=Image.open(image_path))

            for model_item in tuner_table.to_dicts():
                name, metric, path = model_item.values()
                with open(path, "rb") as f_in:
                    model = pickle.loads(f_in.read())
                mlflow.log_metric("f1score", metric)
                if name == "xgb":
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                break

        return None
