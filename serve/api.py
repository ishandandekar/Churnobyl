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
from tagifai import predict

# TODO: Add code for FastAPI and then dockerize
app = fastapi.FastAPI(
    title="Churnobyl", description="Predict the churn, and DO NOT BURN", version="0.1"
)


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(model_dir=config.MODEL_DIR)
    logger.info("Ready for inference!")


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
def predict():
    """
    API endpoint to get predictions for one single data point
    """
    ...


# TODO: and this
def flag():
    """
    API endpoint to flag a prediction. Must contain the predicted label, prediction probability and the actual label
    """
    ...
