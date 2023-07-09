"""
This script contains all the data utility functions.
"""

from pathlib import Path
import requests
import pandas as pd
import pandera as pa
import boto3

# TODO: Add code for schema
SCHEMA = ...


def load_csv_from_dir(dir: Path):
    if dir.is_dir():
        paths = dir.glob("*.csv")
        for path in paths:
            df = pd.read_csv(path)
            # TODO: Validate dataframe
        # TODO: Concat all dfs
    else:
        raise Exception("Path provided is not a directory")


# TODO: Load from url
def load_data_from_url(url: str):
    ...
