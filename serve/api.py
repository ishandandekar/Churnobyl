import json
from datetime import datetime
from functools import wraps
import typing as t
from http import HTTPStatus
import pickle
import fastapi
import wandb
import boto3
from serve.schemas import PredictPayload
from pathlib import Path

# TODO: Add code for FastAPI and then dockerize
app = fastapi.FastAPI(
    title="Churnobyl", description="Predict the churn, and DO NOT BURN", version="0.1"
)


def load_artifacts():
    preprocessor_artifact = wandb.use_artifact(
        "ishandandekar/Churnobyl/churnobyl-ohe-oe-stand:latest", type="preprocessors"
    )
    model_artifact = wandb.use_artifact(
        "ishandandekar/model-registry/churnobyl-binary-clf:latest", type="model"
    )
    preprocessor_path = preprocessor_artifact.download(root=".")
    encoder_oe_path = Path(preprocessor_path) / "encoder_oe_.pkl"
    encoder_ohe_path = Path(preprocessor_path) / "encoder_ohe_.pkl"
    scaler_standard_path = Path(preprocessor_path) / "scaler_standard_.pkl"
    target_encoder_path = Path(preprocessor_path) / "target_encoder_.pkl"
    model_artifact_dir = model_artifact.download(root=".")
    models = list(Path(model_artifact_dir).glob("*.pkl"))
    assert models, "No models found"
    assert len(models) == 1, "More than one model found"
    model = models[0]
    model_path = Path(model_artifact_dir) / model
    artifacts = {
        "model": model_path,
        "encoder_oe": encoder_oe_path,
        "encoder_ohe": encoder_ohe_path,
        "scaler_standard": scaler_standard_path,
        "target_encoder": target_encoder_path,
    }
    return artifacts


def unpickle(artifacts: t.Dict) -> t.Dict:
    """
    Unpickle artifacts
    """
    artifacts = {k: pickle.load(open(v, "rb")) for k, v in artifacts.items()}
    return artifacts


artifacts = load_artifacts()
artifacts = unpickle(artifacts)


def construct_response(f):
    """Construct a JSON reponse for an endpoint"""

    @wraps(f)
    def wrap(request: fastapi.Request, *args, **kwargs) -> t.Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/")
@construct_response
def _index() -> t.Dict:
    """Health check"""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: fastapi.Request, payload: PredictPayload) -> t.Dict:
    """Predict tags for a list of texts."""
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response


# TODO: this
def predict(request: fastapi.Request, payload: PredictPayload) -> t.Dict:
    """
    API endpoint to get predictions for one single data point
    """
    # TODO: Log prediction to S3 bucket
    # TODO: Load Wandb artifacts
    ...


# TODO: and this
def flag():
    """
    API endpoint to flag a prediction. Must contain the predicted label, prediction probability and the actual label
    """
    # TODO: Log flag to S3 bucket
    ...
