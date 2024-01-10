"""
This script contains all the data utility functions.
"""
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import boto3
import cloudpickle as cpickle
import numpy as np
import pandera as pa
import polars as pl
import requests
from box import Box
from scipy.sparse import spmatrix
from sklearn import compose, preprocessing
from sklearn.model_selection import train_test_split

checks: t.Dict[str, t.List[pa.Check]] = {
    "customerID": [],
    "gender": [pa.Check.isin(["Male", "Female"])],
    "SeniorCitizen": [pa.Check.isin([0, 1])],
    "Partner": [pa.Check.isin(["Yes", "No"])],
    "Dependents": [pa.Check.isin(["Yes", "No"])],
    "tenure": [
        pa.Check.greater_than_or_equal_to(min_value=0.0),
    ],
    "PhoneService": [pa.Check.isin(["No", "Yes"])],
    "MultipleLines": [pa.Check.isin(["No", "Yes", "No phone service"])],
    "InternetService": [pa.Check.isin(["DSL", "Fiber optic", "No"])],
    "OnlineSecurity": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "OnlineBackup": [pa.Check.isin(["Yes", "No", "No internet service"])],
    "DeviceProtection": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "TechSupport": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "StreamingTV": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "StreamingMovies": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "Contract": [pa.Check.isin(["Month-to-month", "One year", "Two year"])],
    "PaperlessBilling": [pa.Check.isin(["Yes", "No"])],
    "PaymentMethod": [
        pa.Check.isin(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        )
    ],
    "MonthlyCharges": [],
    "TotalCharges": [],
    "Churn": [pa.Check.isin(["No", "Yes"])],
}


DataSchema = pa.DataFrameSchema(
    columns={
        "customerID": pa.Column(
            dtype="object",
            checks=checks.get("customerID"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "gender": pa.Column(
            dtype="object",
            checks=checks.get("gender"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "SeniorCitizen": pa.Column(
            dtype="int64",
            checks=checks.get("SeniorCitizen"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Partner": pa.Column(
            dtype="object",
            checks=checks.get("Partner"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Dependents": pa.Column(
            dtype="object",
            checks=checks.get("Dependents"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "tenure": pa.Column(
            dtype="int64",
            checks=checks.get("tenure"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "PhoneService": pa.Column(
            dtype="object",
            checks=checks.get("PhoneService"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "MultipleLines": pa.Column(
            dtype="object",
            checks=checks.get("MultipleLines"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "InternetService": pa.Column(
            dtype="object",
            checks=checks.get("InternetService"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "OnlineSecurity": pa.Column(
            dtype="object",
            checks=checks.get("OnlineSecurity"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "OnlineBackup": pa.Column(
            dtype="object",
            checks=checks.get("OnlineBackup"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "DeviceProtection": pa.Column(
            dtype="object",
            checks=checks.get("DeviceProtection"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "TechSupport": pa.Column(
            dtype="object",
            checks=checks.get("TechSupport"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "StreamingTV": pa.Column(
            dtype="object",
            checks=checks.get("StreamingTV"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "StreamingMovies": pa.Column(
            dtype="object",
            checks=checks.get("StreamingMovies"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Contract": pa.Column(
            dtype="object",
            checks=checks.get("Contract"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "PaperlessBilling": pa.Column(
            dtype="object",
            checks=None,
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "PaymentMethod": pa.Column(
            dtype="object",
            checks=checks.get("PaymentMethod"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "MonthlyCharges": pa.Column(
            dtype="float64",
            checks=checks.get("MonthlyCharges"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "TotalCharges": pa.Column(
            dtype="object",
            checks=checks.get("TotalCharges"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Churn": pa.Column(
            dtype="object",
            checks=checks.get("Churn"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
    },
    checks=None,
    index=pa.Index(
        dtype="int64",
        checks=[],
        nullable=False,
        coerce=False,
        name=None,
        description=None,
        title=None,
    ),
    dtype=None,
    coerce=True,
    strict=False,
    name=None,
    ordered=False,
    unique=None,
    report_duplicates="all",
    unique_column_names=False,
    title=None,
    description=None,
)


class BaseDataLoaderStrategy(ABC):
    @abstractmethod
    def __call__(self) -> pl.DataFrame:
        """
        Load data
        """


@dataclass
class DirDataLoaderStrategy(BaseDataLoaderStrategy):
    dir: t.Union[Path, str]

    def __call__(self) -> pl.DataFrame:
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        if self.dir.is_dir():

            def _read(path) -> pl.DataFrame:
                return pl.scan_csv(
                    path, dtypes={"TotalCharges": pl.String}
                ).with_columns(
                    pl.col("TotalCharges")
                    .str.replace(pattern=" ", value="0")
                    .alias("TotalCharges")
                )

            paths = list(self.dir.glob("*.csv"))
            if len(paths) == 0:
                raise Exception(f"No `.csv` present in the directory {dir}")

            return pl.concat(list(map(_read, paths)), how="vertical").collect()
        else:
            raise Exception("Path provided is not a directory")


@dataclass
class UrlDataLoaderStrategy(BaseDataLoaderStrategy):
    url: str

    def __call__(self) -> pl.DataFrame:
        if not self.url.endswith(".csv"):
            raise Exception(f"The url is not of a `.csv`: {self.url}")
        response = requests.get(self.url)
        response.raise_for_status()
        content = StringIO(response.text)

        return pl.read_csv(content, dtypes={"TotalCharges": pl.String}).with_columns(
            pl.col("TotalCharges")
            .str.replace(pattern=" ", value="0")
            .alias("TotalCharges")
        )


@dataclass
class AwsS3DataLoaderStrategy(BaseDataLoaderStrategy):
    bucket_name: str
    folder_path: str
    session: boto3.Session

    def __call__(self) -> pl.DataFrame:
        s3_client = self.session.client("s3")
        if self.folder_path == "":
            s3_url = f"s3://{self.bucket_name}"
            objects = s3_client.list_objects_v2(Bucket=self.bucket_name)
        else:
            s3_url = f"s3://{self.bucket_name}/{self.folder_path}"
            objects = s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.folder_path
            )

        dataframes: list = list()
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv"):
                file_url = f"{s3_url}/{key}"
                try:
                    dataframe = (
                        pl.scan_csv(file_url, dtypes={"TotalCharges": pl.String})
                        .with_columns(
                            pl.col("TotalCharges")
                            .str.replace(pattern=" ", value="0")
                            .alias("TotalCharges")
                        )
                        .collect()
                    )
                    dataframes.append(dataframe)
                    print(f"Loaded: {key}")
                except Exception as e:
                    print(f"Error loading {key}: {e}")

        if len(dataframes) == 0:
            raise Exception(
                "No data found. Check the contents in the bucket and folder"
            )
        return pl.concat(dataframes, ignore_index=True)


DataLoaderStrategyFactory: t.Dict[str, t.Type[BaseDataLoaderStrategy]] = {
    "dir": DirDataLoaderStrategy,
    "url": UrlDataLoaderStrategy,
    "aws_s3": AwsS3DataLoaderStrategy,
}


@dataclass
class TransformerOutput:
    X_train: t.Union[np.ndarray, spmatrix]
    X_test: t.Union[np.ndarray, spmatrix]
    y_train: t.Union[np.ndarray, spmatrix]
    y_test: t.Union[np.ndarray, spmatrix]


class DataEngine:
    @staticmethod
    def load(config: Box) -> pl.DataFrame:
        if config.strategy is not None:
            return DataLoaderStrategyFactory.get(config.strategy)(**config.args)()
        else:
            raise Exception("Arguments for loading the data must be given")

    @staticmethod
    def validate(data: pl.DataFrame) -> pl.DataFrame:
        try:
            DataSchema.validate(data.to_pandas(), lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)
            raise Exception("Schema errors and failure cases")

        return data

    @staticmethod
    def split(
        config: Box,
        data: pl.DataFrame,
        seed: int,
    ) -> t.Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        X, y = data.drop("Churn"), data.select("Churn")
        if config.stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.ratio,
                random_state=seed,
                stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.ratio,
                random_state=seed,
            )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def transform(
        config: Box,
        X_train: pl.DataFrame,
        X_test: pl.DataFrame,
        y_train: pl.DataFrame,
        y_test: pl.DataFrame,
        artifact_dir: Path,
    ) -> TransformerOutput:
        feature_transformer = compose.ColumnTransformer(
            transformers=[
                ("num_scaler", preprocessing.StandardScaler(), config.scale),
                ("cat_ohe", preprocessing.OneHotEncoder(), config.dummy),
                ("cat_oe", preprocessing.OrdinalEncoder(), config.ordinal),
            ]
        )
        label_transformer = preprocessing.LabelBinarizer()
        feature_transformer.fit(X_train.to_pandas())
        label_transformer.fit(y_train.to_pandas())
        preprocessor_paths = ["feature_transformer_.pkl", "label_binarizer.pkl"]
        preprocessors = [feature_transformer, label_transformer]

        def _save_to_pickle(
            path_: str,
            preprocessor: t.Union[
                compose.ColumnTransformer, preprocessing.LabelBinarizer
            ],
        ):
            path: Path = artifact_dir / path_
            with open(str(path), "wb") as f_out:
                cpickle.dump(preprocessor, f_out)

        for path, preprocessor in zip(preprocessor_paths, preprocessors):
            _save_to_pickle(path, preprocessor)
        return TransformerOutput(
            feature_transformer.transform(X_train.to_pandas()),
            feature_transformer.transform(X_test.to_pandas()),
            label_transformer.transform(y_train.to_pandas()),
            label_transformer.transform(y_test.to_pandas()),
        )
