import os
import pickle
import typing as t
import warnings
from http import HTTPStatus
from typing import Union

import fastapi
import mlflow
import pandas as pd
import sklearn
from pydantic import BaseModel, Field

from . import utils

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


class PredictionInputSchema(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)  # Binary field 0 or 1
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)  # Non-negative integer
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
    MonthlyCharges: float = Field(ge=0.0)  # Non-negative float
    TotalCharges: Union[str, float]  # Can be either string or float


def predict_churn(
    data: dict,
    model_pipeline: sklearn.pipeline.Pipeline,
    label_encoder: sklearn.preprocessing.LabelBinarizer,
) -> dict:
    da = {k: [v] for k, v in data.items()}
    df = pd.DataFrame(da)
    pred_label = model_pipeline.predict(df)
    pred_class = label_encoder.inverse_transform(pred_label).tolist()
    pred_prob = model_pipeline.predict_proba(df).tolist()
    return {"prediction_label": pred_class, "prediction_probability": pred_prob}


@app.post("/predict", tags=["Prediction"])
@utils.construct_response
def predict(request: fastapi.Request, data: PredictionInputSchema) -> t.Dict:
    """
    API endpoint to get predictions for one single data point
    """
    result = {}
    result["data"] = data.model_dump()
    result["errors"] = list()
    try:
        prediction_output: dict = predict_churn(
            data.model_dump(), model_pipeline, label_encoder
        )
        result["message"] = prediction_output
        result["status-code"] = HTTPStatus.OK
    except Exception as e:
        result["message"] = HTTPStatus.INTERNAL_SERVER_ERROR.phrase
        result["errors"].append(str(e))
        result["status-code"] = HTTPStatus.INTERNAL_SERVER_ERROR
    return result


# TODO: and this
# @app.post("/flag", tags=["Flagging"])
# def flag(data: utils.FlagEndpointInputSchema):
#     """
#     API endpoint to flag a prediction. Must contain the predicted label, prediction probability and the actual label
#     """
#
#     response = {
#         "message": HTTPStatus.NOT_IMPLEMENTED.phrase,
#         "status-code": HTTPStatus.NOT_IMPLEMENTED,
#         "data": {data.model_dump()},
#     }
#     return response
