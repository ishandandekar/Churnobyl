import random
import typing as t
from pathlib import Path

import numpy as np
import yaml
from box import Box


class Pilot:
    @staticmethod
    def setup(filepath: t.Union[str, Path]) -> t.Tuple[Box, Path, Path, Path, Path]:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, "r") as f_in:
                config = Box(yaml.safe_load(stream=f_in))
            ROOT_DIR = Path.cwd()
            VIZ_DIR: Path = ROOT_DIR / config.PATH.viz
            MODEL_DIR: Path = ROOT_DIR / config.PATH.model
            ARTIFACT_DIR: Path = ROOT_DIR / config.PATH.model / "artifacts"

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
