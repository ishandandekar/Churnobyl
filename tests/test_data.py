"""
Tests for data module in churnobyl.data
"""
import os
import shutil
from pathlib import Path
import boto3
import pandas as pd
from sklearn import datasets
import pytest
from churnobyl.data import DataDreamer


def test_load_csv_from_dir():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    DIR_PATH = "iris_data"
    with pytest.raises(Exception) as e_info:
        _ = DataDreamer.load_csv_from_dir(
            dir=Path(DIR_PATH), columns=iris.feature_names
        )

    os.mkdir(DIR_PATH)

    with pytest.raises(Exception) as e_info:
        _ = DataDreamer.load_csv_from_dir(
            dir=Path(DIR_PATH), columns=iris.feature_names
        )

    _df = df.head(50)
    _ = df.to_csv("iris_data/iris_df.csv", index=False)
    _ = _df.to_csv("iris_data/iris__df.csv", index=False)
    data = DataDreamer.load_csv_from_dir(Path(DIR_PATH), columns=iris.feature_names)

    assert data is not None
    assert data.shape[0] == 200
    assert data.columns.to_list() == iris.feature_names
    shutil.rmtree(DIR_PATH)


def test_load_csv_from_url():
    URL_passes = "https://gist.githubusercontent.com/ishandandekar/545301e423d84f5407af75a482d56b5a/raw/3721bcdb5d4ff091ab5652c2dec25f83b1a0a6b1/toy_dataset.csv"
    URL_fails = "https://abc123.com"
    COLS_passes = ["x", "y"]
    COLS_fails = ["hi", "why are you looking here"]

    with pytest.raises(Exception) as e_info:
        _ = DataDreamer.load_csv_from_url(url=URL_fails, columns=COLS_passes)

    with pytest.raises(Exception) as e_info:
        _ = DataDreamer.load_csv_from_url(url=URL_fails, columns=COLS_fails)

    df = DataDreamer.load_csv_from_url(url=URL_passes, columns=COLS_passes)

    assert df is not None
    assert df.shape[0] > 0


def test_load_csv_from_aws_s3():
    BUCKET_NAME = "churnobyl"
    FOLDER_PATH = ""
    SESSION = boto3.Session()
    _ = DataDreamer.load_csv_from_aws_s3(
        bucket_name=BUCKET_NAME, folder_path=FOLDER_PATH, session=SESSION
    )
    print(_)
