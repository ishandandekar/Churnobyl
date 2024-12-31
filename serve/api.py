import os
import pickle
import typing as t
import warnings
from http import HTTPStatus

import endpoints
import fastapi
import mlflow
import utils

warnings.simplefilter("ignore")

app = fastapi.FastAPI(
    title="Churnobyl", description="Predict the churn, and DO NOT BURN", version="0.1"
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


# TODO: this
@app.post("/predict", tags=["Prediction"])
def predict(resquest: fastapi.Request, data: endpoints.PredictionInputSchema) -> t.Dict:
    """
    API endpoint to get predictions for one single data point
    """
    result = {}
    result["data"] = data.model_dump()
    try:
        prediction_output: dict = endpoints.predict(
            data.model_dump(), model_pipeline, label_encoder
        )
        result["message"] = prediction_output
    except Exception as e:
        result["errors"] = e
    return result


# TODO: and this
@app.post("/flag", tags=["Flagging"])
def flag(data: utils.FlagEndpointInputSchema):
    """
    API endpoint to flag a prediction. Must contain the predicted label, prediction probability and the actual label
    """

    response = {
        "message": HTTPStatus.NOT_IMPLEMENTED.phrase,
        "status-code": HTTPStatus.NOT_IMPLEMENTED,
        "data": {data.model_dump()},
    }
    return response
