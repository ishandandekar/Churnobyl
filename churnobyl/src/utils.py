"""
This script includes all the helper utility functions and classes
"""
import os
import random
import typing as t
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from box import Box
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
    def push_artifacts(tuner: TunerOutput) -> None:
        # Saving tuning results to prefect artifacts
        with pl.Config(fmt_str_lengths=50):
            artifacts.create_table_artifact(
                key="tuning-results",
                table=tuner.as_table().to_dict(as_series=False),
                description="## Tuning results of the run",
            )
        return None
