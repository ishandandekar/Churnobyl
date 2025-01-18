"""Utility/helper functions and variables for api"""

import json
import os
import uuid
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Optional

import fastapi
import mlflow
from google.cloud import storage
from mlflow.tracking import MlflowClient


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
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric_name} DESC"],
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}")
    os.makedirs(local_dir, exist_ok=True)

    if artifact_path:
        client.download_artifacts(
            run_id=best_run.info.run_id, path=artifact_path, dst_path=local_dir
        )
    else:
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
    def wrap(request: fastapi.Request, *args, **kwargs) -> Dict:
        try:
            results: dict = func(request, *args, **kwargs)
        except Exception as e:
            results = {}
            results["message"] = "HOUSTON THERE SEEMS TO BE A PROBLEM"
            results["status-code"] = HTTPStatus.INTERNAL_SERVER_ERROR
            results["errors"] = list()
            results["errors"].append(str(e))
        response = {
            "message": results.get("message", None),
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
            "urlPath": request.url.path,
            "data": results.get("data", None),
            "errors": results.get("errors", list()),
            "IP": request.client.host,
        }
        try:
            upload_response_to_bucket(response)
        except Exception as e:
            response["errors"].append(str(e))
        return response

    return wrap


def upload_response_to_bucket(response: dict):
    if "predict" not in response["urlPath"]:
        return None

    storage_client = storage.Client()
    bucket = storage_client.bucket(os.environ["GCS_BUCKET_NAME"])

    uid = str(uuid.uuid4())
    curr_time = datetime.now().strftime("%H-%M-%S")
    fname = f"{uid}-{curr_time}.json"
    if "predict" in response["urlPath"]:
        fname = "predict/" + fname
    blob = bucket.blob(fname)
    blob.upload_from_string(json.dumps(response))


#
# async def construct_response_and_upload_to_gcloud(request: fastapi.Request, call_next):
#     try:
#         results: dict = await call_next(request)
#     except Exception as e:
#         results = {}
#         results["message"] = "HOUSTON THERE SEEMS TO BE A PROBLEM"
#         results["status-code"] = HTTPStatus.INTERNAL_SERVER_ERROR
#         results["errors"] = list()
#         results["errors"].append(str(e))
#     response = {
#         "message": results.get("message", None),
#         "method": request.method,
#         "status-code": results["status-code"],
#         "timestamp": datetime.now().isoformat(),
#         "url": request.url._url,
#         "urlPath": request.url.path,
#         "data": results.get("data", None),
#         "errors": results.get("errors", list()),
#         "IP": request.client.host,
#     }
#     upload_response_to_bucket(response)
#     return response
