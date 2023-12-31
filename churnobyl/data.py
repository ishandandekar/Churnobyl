"""
This script contains all the data utility functions.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import typing as t
from io import StringIO
from pathlib import Path
import boto3
import pandas as pd
import requests
from pandera import Check, Column, DataFrameSchema, Index


checks: t.Dict[str, t.List[Check]] = {
    "customerID": [],
    "gender": [Check.isin(["Male", "Female"])],
    "SeniorCitizen": [Check.isin([0, 1])],
    "Partner": [Check.isin(["Yes", "No"])],
    "Dependents": [Check.isin(["Yes", "No"])],
    "tenure": [
        Check.greater_than_or_equal_to(min_value=0.0),
    ],
    "PhoneService": [Check.isin(["No", "Yes"])],
    "MultipleLines": [Check.isin(["No", "Yes", "No phone service"])],
    "InternetService": [Check.isin(["DSL", "Fiber optic", "No"])],
    "OnlineSecurity": [Check.isin(["No", "Yes", "No internet service"])],
    "OnlineBackup": [Check.isin(["Yes", "No", "No internet service"])],
    "DeviceProtection": [Check.isin(["No", "Yes", "No internet service"])],
    "TechSupport": [Check.isin(["No", "Yes", "No internet service"])],
    "StreamingTV": [Check.isin(["No", "Yes", "No internet service"])],
    "StreamingMovies": [Check.isin(["No", "Yes", "No internet service"])],
    "Contract": [Check.isin(["Month-to-month", "One year", "Two year"])],
    "PaperlessBilling": [Check.isin(["Yes", "No"])],
    "PaymentMethod": [
        Check.isin(
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
    "Churn": [Check.isin(["No", "Yes"])],
}

CHURN_SCHEMA = DataFrameSchema(
    columns={
        "Churn": Column(
            dtype="object",
            checks=checks.get("Churn"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        )
    },
    checks=None,
    index=Index(
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


INPUT_SCHEMA = DataFrameSchema(
    columns={
        "customerID": Column(
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
        "gender": Column(
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
        "SeniorCitizen": Column(
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
        "Partner": Column(
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
        "Dependents": Column(
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
        "tenure": Column(
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
        "PhoneService": Column(
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
        "MultipleLines": Column(
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
        "InternetService": Column(
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
        "OnlineSecurity": Column(
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
        "OnlineBackup": Column(
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
        "DeviceProtection": Column(
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
        "TechSupport": Column(
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
        "StreamingTV": Column(
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
        "StreamingMovies": Column(
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
        "Contract": Column(
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
        "PaperlessBilling": Column(
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
        "PaymentMethod": Column(
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
        "MonthlyCharges": Column(
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
        "TotalCharges": Column(
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
    },
    checks=None,
    index=Index(
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


TRAINING_SCHEMA = DataFrameSchema(
    columns={
        "customerID": Column(
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
        "gender": Column(
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
        "SeniorCitizen": Column(
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
        "Partner": Column(
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
        "Dependents": Column(
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
        "tenure": Column(
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
        "PhoneService": Column(
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
        "MultipleLines": Column(
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
        "InternetService": Column(
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
        "OnlineSecurity": Column(
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
        "OnlineBackup": Column(
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
        "DeviceProtection": Column(
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
        "TechSupport": Column(
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
        "StreamingTV": Column(
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
        "StreamingMovies": Column(
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
        "Contract": Column(
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
        "PaperlessBilling": Column(
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
        "PaymentMethod": Column(
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
        "MonthlyCharges": Column(
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
        "TotalCharges": Column(
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
        "Churn": Column(
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
    index=Index(
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


class DataLoaderStrategy(ABC):
    @abstractmethod
    def __call__(self) -> pd.DataFrame:
        """
        Load data
        """


@dataclass
class DirDataLoaderStrategy(DataLoaderStrategy):
    dir: t.Union[Path, str]
    columns: list[str]

    def __call__(self) -> pd.DataFrame:
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        if self.dir.is_dir():
            data = pd.DataFrame(columns=self.columns)
            paths = list(self.dir.glob("*.csv"))
            if len(paths) == 0:
                raise Exception(f"No `.csv` present in the directory {dir}")
            for path in paths:
                df = pd.read_csv(path)
                data = pd.concat([data, df])
            return data
        else:
            raise Exception("Path provided is not a directory")


@dataclass
class UrlDataLoaderStrategy(DataLoaderStrategy):
    url: str
    columns: list[str]

    def __call__(self) -> pd.DataFrame:
        if not self.url.endswith(".csv"):
            raise Exception(f"The url is not of a `.csv`: {self.url}")
        response = requests.get(self.url)
        response.raise_for_status()
        csv_file = StringIO(response.text)
        df = pd.read_csv(csv_file)
        if df.columns.to_list() != self.columns:
            raise Exception(f"Column mismatch error for `.csv`: {self.url}")
        return df


@dataclass
class AwsS3DataLoaderStrategy(DataLoaderStrategy):
    bucket_name: str
    folder_path: str
    session: boto3.Session

    def __call__(self) -> pd.DataFrame:
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
                print(file_url)
                try:
                    dataframe = pd.read_csv(file_url)
                    dataframes.append(dataframe)
                    print(f"Loaded: {key}")
                except Exception as e:
                    print(f"Error loading {key}: {e}")

        if len(dataframes) == 0:
            raise Exception(
                "No data found. Check the contents in the bucket and folder"
            )
        return pd.concat(dataframes, ignore_index=True)


DataLoaderStrategyFactory: t.Dict[str, t.Type[DataLoaderStrategy]] = {
    "dir": DirDataLoaderStrategy,
    "url": UrlDataLoaderStrategy,
    "aws_s3": AwsS3DataLoaderStrategy,
}
