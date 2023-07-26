from datetime import datetime
from functools import wraps
import typing as t
import uuid
import json
from http import HTTPStatus
import pickle
import fastapi
import wandb
import boto3
from serve.schemas import PredictPayload
from pathlib import Path
from pandera import Check, Column, DataFrameSchema, Index, MultiIndex
import pandas as pd
import pandera as pa
from munch import Munch
import yaml
from pydantic import BaseModel

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
    "Churn": [Check.isin(["No", "Yes"])],
    "actualChurn": [],
    "predictionChurn": [],
    "predicted_probaChurn": [],
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

FLAG_SCHEMA = DataFrameSchema(
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


class PredEndpointInputSchema(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
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
    tenure: float
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


def set_config(config_path: Path) -> Munch:
    if config_path.exists():
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                return Munch(config)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        raise Exception("Path error occured. File does not exist")


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


config = set_config("./serve-config.yaml")
artifacts = load_artifacts()
artifacts = unpickle(artifacts)

# TODO: Add code for FastAPI and then dockerize
app = fastapi.FastAPI(
    title="Churnobyl", description="Predict the churn, and DO NOT BURN", version="0.1"
)


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
def predict(input_data: PredEndpointInputSchema) -> t.Dict:
    """
    API endpoint to get predictions for one single data point
    """
    # TODO: Log prediction to S3 bucket
    inputs = {
        "customerID": input_data.customerID,
        "gender": input_data.gender,
        "SeniorCitizen": input_data.SeniorCitizen,
        "Partner": input_data.Partner,
        "Dependents": input_data.Dependents,
        "tenure": input_data.tenure,
        "PhoneService": input_data.PhoneService,
        "MultipleLines": input_data.MultipleLines,
        "InternetService": input_data.InternetService,
        "OnlineSecurity": input_data.OnlineSecurity,
        "OnlineBackup": input_data.OnlineBackup,
        "DeviceProtection": input_data.DeviceProtection,
        "TechSupport": input_data.TechSupport,
        "StreamingTV": input_data.StreamingTV,
        "StreamingMovies": input_data.StreamingMovies,
        "Contract": input_data.Contract,
        "PaperlessBilling": input_data.PaperlessBilling,
        "PaymentMethod": input_data.PaymentMethod,
        "MonthlyCharges": input_data.MonthlyCharges,
        "TotalCharges": input_data.TotalCharges,
    }
    input_df = pd.DataFrame(inputs)
    response = {}
    response["errors"] = {}
    try:
        INPUT_SCHEMA.validate(input_df)
    except pa.errors.SchemaError as err:
        response["errors"]["schema_failure_cases"] = err.failure_cases
        response["errors"]["data"] = err.data
    input_df["TotalCharges"] = input_df["TotalCharges"].replace(
        to_replace=" ", value="0"
    )
    input_ohe = input_df[config.data.get("CAT_COLS_OHE")]
    input_ohe_trans = artifacts["encoder_ohe"].transform(input_ohe)
    input_ohe_trans__df = pd.DataFrame(
        input_ohe_trans.toarray(),
        columns=artifacts["encoder_ohe"].get_feature_names_out(),
    )
    input_oe = input_df[config.data.get("CAT_COLS_OE")]
    input_oe_trans: pd.DataFrame = artifacts["encoder_oe"].transform(input_oe)
    input_scale = input_df[config.data.get("NUM_COLS")]
    input_scale_trans: pd.DataFrame = artifacts["scaler_standard"].transform(
        input_scale
    )
    input_to_predict = pd.concat(
        [
            input_ohe_trans__df.reset_index(drop=True),
            input_oe_trans.reset_index(drop=True),
            input_scale_trans.reset_index(drop=True),
        ],
        axis=1,
    )
    prediction = artifacts["model"].predict(input_to_predict)
    prediction_proba = artifacts["model"].predict_proba(input_to_predict)
    response["prediction"] = prediction
    response["prediction_proba"] = prediction_proba
    ...


# TODO: and this
@app.post("/flag", tags=["Flagging"])
def flag(flag_data: FlagEndpointInputSchema):
    """
    API endpoint to flag a prediction. Must contain the predicted label, prediction probability and the actual label
    """
    flag_data = {
        "customerID": flag_data.customerID,
        "gender": flag_data.gender,
        "SeniorCitizen": flag_data.SeniorCitizen,
        "Partner": flag_data.Partner,
        "Dependents": flag_data.Dependents,
        "tenure": flag_data.tenure,
        "PhoneService": flag_data.PhoneService,
        "MultipleLines": flag_data.MultipleLines,
        "InternetService": flag_data.InternetService,
        "OnlineSecurity": flag_data.OnlineSecurity,
        "OnlineBackup": flag_data.OnlineBackup,
        "DeviceProtection": flag_data.DeviceProtection,
        "TechSupport": flag_data.TechSupport,
        "StreamingTV": flag_data.StreamingTV,
        "StreamingMovies": flag_data.StreamingMovies,
        "Contract": flag_data.Contract,
        "PaperlessBilling": flag_data.PaperlessBilling,
        "PaymentMethod": flag_data.PaymentMethod,
        "MonthlyCharges": flag_data.MonthlyCharges,
        "TotalCharges": flag_data.TotalCharges,
        "actualChurn": flag_data.actualChurn,
        "predictedChurn": flag_data.predictedChurn,
        "predicted_probaChurn": flag_data.predicted_probaChurn,
    }
    flag_df = pd.DataFrame(flag_data)
    response = dict()
    try:
        FLAG_SCHEMA.validate(flag_df)
        s3 = boto3.client("s3")

        # Generate a unique file name
        file_name = f"{str(uuid.uuid4())}.json"
        json_string = json.dumps(flag_data)
        try:
            s3_response = s3.put_object(
                Body=json_string,
                Bucket=config.s3.get("BUCKET_NAME"),
                Key="flagged/" + file_name,
            )
            if s3_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                response["message"] = "Flagged successfully"
                response["status-code"] = HTTPStatus.OK
                response["data"] = {"file_name": file_name}
            else:
                response["message"] = "Failed to upload `.json` file to S3"
                response["status-code"] = HTTPStatus.INTERNAL_SERVER_ERROR
                response["data"] = {}
        except Exception as e:
            print(e)
    except pa.errors.SchemaError as err:
        response["errors"]["schema_failure_cases"] = err.failure_cases
        response["errors"]["data"] = err.data
