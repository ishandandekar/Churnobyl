import os
import pickle
import typing as t
import warnings
from http import HTTPStatus

import fastapi
import mlflow

from . import predict, utils

warnings.simplefilter("ignore")

app = fastapi.FastAPI(
    title="Churnobyl", description="Predict the churn, and DO NOT BURN", version="0.2"
)


MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_EXPERIMENT_TRACKING_URI")
ARTIFACT_DIR = "best_run_artifacts"
utils.fetch_best_run_artifacts(
    experiment_name=MLFLOW_EXPERIMENT_NAME,
    tracking_uri=MLFLOW_TRACKING_URI,
    local_dir=ARTIFACT_DIR,
)
model_pipeline = mlflow.sklearn.load_model(ARTIFACT_DIR + "/model")
label_encoder = pickle.load(open(ARTIFACT_DIR + "/label_binarizer.pkl", "rb"))


@app.get("/")
@utils.construct_response
def _index(request: fastapi.Request) -> t.Dict:
    """Health check"""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict", tags=["Prediction"])
@utils.construct_response
def predict(request: fastapi.Request, data: predict.PredictionInputSchema) -> t.Dict:
    """
    API endpoint to get predictions for one single data point
    """
    result = {}
    result["data"] = data.model_dump()
    result["errors"] = list()
    try:
        prediction_output: dict = predict.predict_churn(
            data.model_dump(), model_pipeline, label_encoder
        )
        result["message"] = prediction_output
        result["status-code"] = HTTPStatus.OK
    except Exception as e:
        result["message"] = HTTPStatus.INTERNAL_SERVER_ERROR.phrase
        result["errors"].append(str(e))
        result["status-code"] = HTTPStatus.INTERNAL_SERVER_ERROR
    return result
