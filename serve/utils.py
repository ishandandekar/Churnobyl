"""
Utility/helper functions and variables for api
"""

import os
import typing as t
from datetime import datetime
from functools import wraps
from typing import Optional

import fastapi
import mlflow
from mlflow.tracking import MlflowClient

# from pandera import Check, Column, DataFrameSchema, Index
from pydantic import BaseModel

# checks: t.Dict[str, t.List[Check]] = {
#     "customerID": [],
#     "gender": [Check.isin(["Male", "Female"])],
#     "SeniorCitizen": [Check.isin([0, 1])],
#     "Partner": [Check.isin(["Yes", "No"])],
#     "Dependents": [Check.isin(["Yes", "No"])],
#     "tenure": [
#         Check.greater_than_or_equal_to(min_value=0.0),
#         Check.less_than_or_equal_to(max_value=72.0),
#     ],
#     "PhoneService": [Check.isin(["No", "Yes"])],
#     "MultipleLines": [Check.isin(["No", "Yes", "No phone service"])],
#     "InternetService": [Check.isin(["DSL", "Fiber optic", "No"])],
#     "OnlineSecurity": [Check.isin(["No", "Yes", "No internet service"])],
#     "OnlineBackup": [Check.isin(["Yes", "No", "No internet service"])],
#     "DeviceProtection": [Check.isin(["No", "Yes", "No internet service"])],
#     "TechSupport": [Check.isin(["No", "Yes", "No internet service"])],
#     "StreamingTV": [Check.isin(["No", "Yes", "No internet service"])],
#     "StreamingMovies": [Check.isin(["No", "Yes", "No internet service"])],
#     "Contract": [Check.isin(["Month-to-month", "One year", "Two year"])],
#     "PaperlessBilling": [Check.isin(["Yes", "No"])],
#     "PaymentMethod": [
#         Check.isin(
#             [
#                 "Electronic check",
#                 "Mailed check",
#                 "Bank transfer (automatic)",
#                 "Credit card (automatic)",
#             ]
#         )
#     ],
#     "MonthlyCharges": [],
#     "TotalCharges": [],
#     "TrueChurn": [Check.isin(["No", "Yes"])],
#     "PredictedChurn": [Check.isin(["No", "Yes"])],
#     "PredictedChurnProbability": [],
# }

# INPUT_SCHEMA = DataFrameSchema(
#     columns={
#         "customerID": Column(
#             dtype="object",
#             checks=checks.get("customerID"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "gender": Column(
#             dtype="object",
#             checks=checks.get("gender"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "SeniorCitizen": Column(
#             dtype="int64",
#             checks=checks.get("SeniorCitizen"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "Partner": Column(
#             dtype="object",
#             checks=checks.get("Partner"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "Dependents": Column(
#             dtype="object",
#             checks=checks.get("Dependents"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "tenure": Column(
#             dtype="int64",
#             checks=checks.get("tenure"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "PhoneService": Column(
#             dtype="object",
#             checks=checks.get("PhoneService"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "MultipleLines": Column(
#             dtype="object",
#             checks=checks.get("MultipleLines"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "InternetService": Column(
#             dtype="object",
#             checks=checks.get("InternetService"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "OnlineSecurity": Column(
#             dtype="object",
#             checks=checks.get("OnlineSecurity"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "OnlineBackup": Column(
#             dtype="object",
#             checks=checks.get("OnlineBackup"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "DeviceProtection": Column(
#             dtype="object",
#             checks=checks.get("DeviceProtection"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "TechSupport": Column(
#             dtype="object",
#             checks=checks.get("TechSupport"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "StreamingTV": Column(
#             dtype="object",
#             checks=checks.get("StreamingTV"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "StreamingMovies": Column(
#             dtype="object",
#             checks=checks.get("StreamingMovies"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "Contract": Column(
#             dtype="object",
#             checks=checks.get("Contract"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "PaperlessBilling": Column(
#             dtype="object",
#             checks=None,
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "PaymentMethod": Column(
#             dtype="object",
#             checks=checks.get("PaymentMethod"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "MonthlyCharges": Column(
#             dtype="float64",
#             checks=checks.get("MonthlyCharges"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "TotalCharges": Column(
#             dtype="object",
#             checks=checks.get("TotalCharges"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#     },
#     checks=None,
#     index=Index(
#         dtype="int64",
#         checks=[],
#         nullable=False,
#         coerce=False,
#         name=None,
#         description=None,
#         title=None,
#     ),
#     dtype=None,
#     coerce=True,
#     strict=False,
#     name=None,
#     ordered=False,
#     unique=None,
#     report_duplicates="all",
#     unique_column_names=False,
#     title=None,
#     description=None,
# )

# FLAG_SCHEMA: pa.DataFrameSchema = DataFrameSchema(
#     columns={
#         "customerID": Column(
#             dtype="object",
#             checks=checks.get("customerID"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "gender": Column(
#             dtype="object",
#             checks=checks.get("gender"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "SeniorCitizen": Column(
#             dtype="int64",
#             checks=checks.get("SeniorCitizen"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "Partner": Column(
#             dtype="object",
#             checks=checks.get("Partner"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "Dependents": Column(
#             dtype="object",
#             checks=checks.get("Dependents"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "tenure": Column(
#             dtype="int64",
#             checks=checks.get("tenure"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "PhoneService": Column(
#             dtype="object",
#             checks=checks.get("PhoneService"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "MultipleLines": Column(
#             dtype="object",
#             checks=checks.get("MultipleLines"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "InternetService": Column(
#             dtype="object",
#             checks=checks.get("InternetService"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "OnlineSecurity": Column(
#             dtype="object",
#             checks=checks.get("OnlineSecurity"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "OnlineBackup": Column(
#             dtype="object",
#             checks=checks.get("OnlineBackup"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "DeviceProtection": Column(
#             dtype="object",
#             checks=checks.get("DeviceProtection"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "TechSupport": Column(
#             dtype="object",
#             checks=checks.get("TechSupport"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "StreamingTV": Column(
#             dtype="object",
#             checks=checks.get("StreamingTV"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "StreamingMovies": Column(
#             dtype="object",
#             checks=checks.get("StreamingMovies"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "Contract": Column(
#             dtype="object",
#             checks=checks.get("Contract"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "PaperlessBilling": Column(
#             dtype="object",
#             checks=None,
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "PaymentMethod": Column(
#             dtype="object",
#             checks=checks.get("PaymentMethod"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "MonthlyCharges": Column(
#             dtype="float64",
#             checks=checks.get("MonthlyCharges"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "TotalCharges": Column(
#             dtype="object",
#             checks=checks.get("TotalCharges"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "actualChurn": Column(
#             dtype="int64",
#             checks=checks.get("actualChurn"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "predictedChurn": Column(
#             dtype="int64",
#             checks=checks.get("predictedChurn"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#         "predicted_probaChurn": Column(
#             dtype="float64",
#             checks=checks.get("predicted_probaChurn"),
#             nullable=False,
#             unique=False,
#             coerce=False,
#             required=True,
#             regex=False,
#             description=None,
#             title=None,
#         ),
#     },
#     checks=None,
#     index=Index(
#         dtype="int64",
#         checks=[],
#         nullable=False,
#         coerce=False,
#         name=None,
#         description=None,
#         title=None,
#     ),
#     dtype=None,
#     coerce=True,
#     strict=False,
#     name=None,
#     ordered=False,
#     unique=None,
#     report_duplicates="all",
#     unique_column_names=False,
#     title=None,
#     description=None,
# )


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


def fetch_best_run_artifacts(
    experiment_name: str,
    metric_name: str = "f1_score",
    tracking_uri: Optional[str] = None,
    artifact_path: Optional[str] = None,
    local_dir: str = "downloaded_artifacts",
) -> None:
    """
    Fetch artifacts from the MLflow run with the highest specified metric value.

    Args:
        experiment_name (str): Name of the MLflow experiment
        metric_name (str): Name of the metric to sort by (default: "f1_score")
        tracking_uri (Optional[str]): MLflow tracking server URI
        artifact_path (Optional[str]): Specific artifact path to download. If None, downloads all artifacts
        local_dir (str): Local directory to save artifacts to
    """
    # Set up MLflow client
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Get all runs for the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric_name} DESC"],
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    # Get the best run (first run since we ordered by metric DESC)
    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}")
    # print(f"Best {metric_name}: {
    #       best_run.data.metrics.get(metric_name, 'N/A')}")
    #
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Download artifacts
    if artifact_path:
        # Download specific artifact or directory
        client.download_artifacts(
            run_id=best_run.info.run_id, path=artifact_path, dst_path=local_dir
        )
        # print(
        #     f"Downloaded artifact(s) from path '{
        #         artifact_path}' to '{local_dir}'"
        # )
    else:
        # Download all artifacts
        artifacts = client.list_artifacts(best_run.info.run_id)
        for artifact in artifacts:
            client.download_artifacts(
                run_id=best_run.info.run_id, path=artifact.path, dst_path=local_dir
            )
        print(f"Downloaded all artifacts to '{local_dir}'")


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
