import random
import typing as t
from pathlib import Path

import numpy as np
import yaml
from box import Box
from exceptions import ConfigValidationError

from data import validate_data_config
from model import validate_model_config


class Pilot:
    @staticmethod
    def __validate_config(config: Box):
        if not isinstance(config.get("SEED"), int):
            raise ConfigValidationError("SEED should be an integer.")
        validate_data_config(config=config)
        validate_model_config(config=config)

    @staticmethod
    def setup(filepath: t.Union[str, Path]) -> t.Tuple[Box, Path, Path, Path, Path]:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, "r") as f_in:
                config = Box(yaml.safe_load(stream=f_in))

            Pilot.__validate_config(config=config)

            ROOT_DIR = Path.cwd()
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


if __name__ == "__main__":
    tup = Pilot.setup("./churnobyl/conf/config.yaml")
