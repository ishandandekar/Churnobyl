"""
This script contains all the data utility functions.
"""

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
            checks=None,
            nullable=checks.get("OnlineBackup"),
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


# DEV: Change file format other than `.csv` here
class DataDreamer:
    @staticmethod
    def load_csv_from_dir(dir: Path, columns: list) -> pd.DataFrame:
        if dir.is_dir():
            data = pd.DataFrame(columns=columns)
            paths = list(dir.glob("*.csv"))
            if len(paths) == 0:
                raise Exception(f"No `.csv` present in the directory {dir}")
            for path in paths:
                df = pd.read_csv(path)
                # if data.columns != df.columns:
                #     raise Exception(f"Column mismatch error for `.csv`: {path}")
                data = pd.concat([data, df])
            return data
        else:
            raise Exception("Path provided is not a directory")

    @staticmethod
    def load_csv_from_url(url: str, columns: list) -> pd.DataFrame:
        if not url.endswith(".csv"):
            raise Exception(f"The url is not of a `.csv`: {url}")
        response = requests.get(url)
        response.raise_for_status()
        csv_file = response.content
        df = pd.read_csv(pd.compat.StringIO(csv_file.decode("utf-8")))
        if df.columns.to_list() != columns:
            raise Exception(f"Column mismatch error for `.csv`: {url}")
        return df

    @staticmethod
    def load_csv_from_aws_s3(
        bucket_name: str, region: str, session: boto3.Session
    ) -> pd.DataFrame:
        # How to create a session?
        # session = boto3.Session(
        #     aws_access_key_id='your-access-key',
        #     aws_secret_access_key='your-secret-key',
        #     region_name='your-aws-region'
        # )
        s3_client = session.client("s3")
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        dataframes: list = list()
        for obj in response["Contents"]:
            key: str = obj["Key"]
            if key.endswith(".csv"):
                response = requests.get(
                    f"https://{bucket_name}.s3.{region}.amazonaws.com/{key}"
                )
                response.raise_for_status()
                csv_content = response.content.decode("utf-8")

                df = pd.read_csv(StringIO(csv_content))

                dataframes.append(df)
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
