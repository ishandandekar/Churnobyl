import fastapi
import uvicorn
from http import HTTPStatus
from functools import wraps
import typing as t
from pydantic import BaseModel
from pandera import Check, Column, DataFrameSchema, Index
import pandas as pd
import pandera as pa
from datetime import datetime
import wandb
from pathlib import Path
import yaml
from munch import Munch
import os
import pickle as pkl
from sklearn import preprocessing, ensemble
import xgboost as xgb
import warnings

warnings.simplefilter("ignore")

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


def construct_response(f):
    """
    Construct a JSON response for and endpoint
    """

    @wraps(f)
    def wrap(request: fastapi.Request, *args, **kwargs) -> t.Dict:
        results: dict = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
            "data": results.get("data", None),
            "errors": results.get("errors", None),
            "prediction": results.get("prediction", None),
            "prediction_proba": results.get("prediction_proba", None),
        }
        return response

    return wrap


def set_config(config_path: Path, wandb_api_key: str) -> Munch:
    if config_path.exists():
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                wandb.login(key=wandb_api_key)
                return Munch(config)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        raise Exception("Path error occured. File does not exist")


def _custom_combiner(feature, category) -> str:
    """
    Creates custom column name for every category

    Args:
        feature (str): column name
        category (str): name of the category

    Returns:
        str: column name
    """
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)


def load_artifacts() -> t.Dict:
    wand_api = wandb.Api()
    preprocessor_artifact = wand_api.artifact(
        "ishandandekar/Churnobyl/churnobyl-ohe-oe-stand:latest", type="preprocessors"
    )
    model_artifact = wand_api.artifact(
        "ishandandekar/model-registry/churnobyl-binary-clf:latest", type="model"
    )
    preprocessor_path = preprocessor_artifact.download(root=".")
    encoder_oe_path = Path(preprocessor_path) / "encoder_oe_.pkl"
    encoder_ohe_path = Path(preprocessor_path) / "encoder_ohe_.pkl"
    scaler_standard_path = Path(preprocessor_path) / "scaler_standard_.pkl"
    target_encoder_path = Path(preprocessor_path) / "target_encoder_.pkl"
    model_artifact_dir = model_artifact.download(root=".")
    models: t.List[Path] = list(Path(model_artifact_dir).glob("*best_.pkl"))
    # assert models, "No models found"
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


def unpickle_artifacts(artifacts: t.Dict) -> t.Dict:
    """
    Unpickle artifacts
    """
    artifacts: t.Dict = {k: pkl.load(open(v, "rb")) for k, v in artifacts.items()}
    return artifacts


CONFIG_PATH = Path("./temp-serve-config.yaml")
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
config = set_config(config_path=CONFIG_PATH, wandb_api_key=WANDB_API_KEY)
artifacts = load_artifacts()
artifacts = unpickle_artifacts(artifacts=artifacts)

app = fastapi.FastAPI(title="TestAPI", description="idkdkkdkdkd", version="1.0")


@app.get("/")
@construct_response
def _index(request: fastapi.Request) -> t.Dict:
    """Health check"""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/predict")
@construct_response
def predict(
    request: fastapi.Request, input_data: PredictionEndpointInputSchema
) -> t.Dict:
    "Predict on a data-point"

    response: t.Dict = dict()

    input_df = {
        "customerID": [input_data.customerID],
        "gender": [input_data.gender],
        "SeniorCitizen": [input_data.SeniorCitizen],
        "Partner": [input_data.Partner],
        "Dependents": [input_data.Dependents],
        "tenure": [input_data.tenure],
        "PhoneService": [input_data.PhoneService],
        "MultipleLines": [input_data.MultipleLines],
        "InternetService": [input_data.InternetService],
        "OnlineSecurity": [input_data.OnlineSecurity],
        "OnlineBackup": [input_data.OnlineBackup],
        "DeviceProtection": [input_data.DeviceProtection],
        "TechSupport": [input_data.TechSupport],
        "StreamingTV": [input_data.StreamingTV],
        "StreamingMovies": [input_data.StreamingMovies],
        "Contract": [input_data.Contract],
        "PaperlessBilling": [input_data.PaperlessBilling],
        "PaymentMethod": [input_data.PaymentMethod],
        "MonthlyCharges": [input_data.MonthlyCharges],
        "TotalCharges": [input_data.TotalCharges],
    }
    input_df = pd.DataFrame.from_dict(input_df)
    response["data"] = input_df.to_json(orient="records", index=False)
    try:
        INPUT_SCHEMA.validate(input_df, lazy=True)

        X_ohe = input_df[config.data.get("CAT_COLS_OHE")].values
        X_ohe = artifacts.get("encoder_ohe").transform(X_ohe)
        X_ohe = pd.DataFrame(
            X_ohe.toarray(),
            columns=artifacts.get("encoder_ohe").get_feature_names_out(),
        )

        X_oe = input_df[config.data.get("CAT_COLS_OE")].values
        X_oe = artifacts.get("encoder_oe").transform(X_oe)
        X_oe = pd.DataFrame(X_oe, columns=config.data.get("CAT_COLS_OE"))

        X_scale = input_df[config.data.get("NUM_COLS")].values
        X_scale = artifacts.get("scaler_standard").transform(X_scale)
        X_scale = pd.DataFrame(X_scale, columns=config.data.get("NUM_COLS"))

        X = pd.concat(
            [
                X_ohe.reset_index(drop=True),
                X_oe.reset_index(drop=True),
                X_scale.reset_index(drop=True),
            ],
            axis=1,
        )
        prediction_proba = max(
            artifacts.get("model").predict_proba(X.values).squeeze().tolist()
        )
        prediction = artifacts.get("model").predict(X.values).squeeze().tolist()

        response["message"] = HTTPStatus.OK.phrase
        response["status-code"] = HTTPStatus.OK
        response["prediction"] = prediction
        response["prediction_proba"] = prediction_proba

    except pa.errors.SchemaErrors as err:
        response["message"] = HTTPStatus.BAD_REQUEST.phrase
        response["status-code"] = HTTPStatus.BAD_REQUEST
        response["errors"] = err

    return response


def main():
    uvicorn.run(app=app, port=8000, host="0.0.0.0")


if __name__ == "__main__":
    main()
