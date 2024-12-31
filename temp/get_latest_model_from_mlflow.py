import os
from typing import List, Optional

import mlflow
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


def list_available_artifacts(
    experiment_name: str, tracking_uri: Optional[str] = None
) -> List[str]:
    """
    List all available artifacts in the best run.

    Args:
        experiment_name (str): Name of the MLflow experiment
        tracking_uri (Optional[str]): MLflow tracking server URI

    Returns:
        List[str]: List of artifact paths
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
        order_by=["metrics.f1_score DESC"],
    )

    if not runs:
        return []

    artifacts = client.list_artifacts(runs[0].info.run_id)
    return [artifact.path for artifact in artifacts]


# Example usage
if __name__ == "__main__":
    # Example configuration
    EXPERIMENT_NAME = "churnobyl"
    # Replace with your MLflow server URI
    MLFLOW_TRACKING_URI = "http://localhost:5000"

    try:
        # List available artifacts first
        print("Available artifacts:")
        artifacts = list_available_artifacts(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=MLFLOW_TRACKING_URI,
        )
        for artifact in artifacts:
            print(f"- {artifact}")
        print("Fetching now")
        # Download all artifacts from the best run
        fetch_best_run_artifacts(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=MLFLOW_TRACKING_URI,
            local_dir="best_run_artifacts",
        )

        # Or download specific artifacts
        # fetch_best_run_artifacts(
        #     experiment_name=EXPERIMENT_NAME,
        #     tracking_uri=MLFLOW_TRACKING_URI,
        #     artifact_path="model",
        #     local_dir="best_model"
        # )

    except Exception as e:
        print(f"Error: {str(e)}")
