"""
Utility/helper functions and variables for api
"""
from pathlib import Path
import pickle as pkl
from datetime import datetime
import typing as t
from functools import wraps
from pandera import Check, Column, DataFrameSchema, Index
import pandas as pd
import pandera as pa
from pydantic import BaseModel
import fastapi
import wandb
import yaml

checks: t.Dict[str, t.List[Check]] = {
    "customerID": [],
    "gender": [Check.isin(["Male", "Female"])],
    "SeniorCitizen": [Check.isin([0, 1])],
    "Partner": [Check.isin(["Yes", "No"])],
    "Dependents": [Check.isin(["Yes", "No"])],
    "tenure": [
        Check.greater_than_or_equal_to(min_value=0.0),
        Check.less_than_or_equal_to(max_value=72.0),
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
    "TrueChurn": [Check.isin(["No", "Yes"])],
    "PredictedChurn": [Check.isin(["No", "Yes"])],
    "PredictedChurnProbability": [],
}

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

FLAG_SCHEMA: pa.DataFrameSchema = DataFrameSchema(
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
        "actualChurn": Column(
            dtype="int64",
            checks=checks.get("actualChurn"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "predictedChurn": Column(
            dtype="int64",
            checks=checks.get("predictedChurn"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "predicted_probaChurn": Column(
            dtype="float64",
            checks=checks.get("predicted_probaChurn"),
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


class PredictionEndpointInputSchema(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


class FlagEndpointInputSchema(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str
    actualChurn: int
    predictedChurn: int
    predicted_probaChurn: float


def construct_response(func):
    """
    Construct a JSON response for and endpoint
    """

    @wraps(func)
    def wrap(request: fastapi.Request, *args, **kwargs) -> t.Dict:
        results: dict = func(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
            "data": results.get("data", None),
            "errors": results.get("errors", None),
            "IP": request.client.host,
        }
        return response

    return wrap


def set_config(config_path: Path, WANDB_API_KEY: str) -> Munch:
    if config_path.exists():
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                wandb.login(key=WANDB_API_KEY)
                return Munch(config)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        raise Exception("Path error occured. File does not exist")


def unpickle_artifacts(artifacts: t.Dict) -> t.Dict:
    """
    Unpickle artifacts
    """
    artifacts: t.Dict = {k: pkl.load(open(v, "rb")) for k, v in artifacts.items()}
    return artifacts
