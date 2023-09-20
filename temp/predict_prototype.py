from datetime import datetime
import os
from functools import wraps
import typing as t
import uuid
import json
from http import HTTPStatus
import pickle
import fastapi
import wandb
import boto3
from pathlib import Path
from pandera import Check, Column, DataFrameSchema, Index, MultiIndex
import pandera as pa
import pandas as pd
import pandera as pa
from munch import Munch
import yaml
from pydantic import BaseModel
import uvicorn
from sklearn import preprocessing, ensemble
import warnings
import xgboost as xgb

warnings.simplefilter("ignore")


def set_config(config_path: Path, WANDB_API_KEY: str) -> Munch:
    if config_path.exists():
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                wandb.login(key=WANDB_API_KEY)
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


def load_artifacts():
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
    models = list(Path(model_artifact_dir).glob("*.pkl"))
    # assert models, "No models found"
    # assert len(models) == 1, "More than one model found"
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


# CONFIG_PATH = Path("./serve/serve-config.yaml")
# WANDB_API_KEY = ...
# config = set_config(config_path=CONFIG_PATH, WANDB_API_KEY=WANDB_API_KEY)
# artifacts = load_artifacts()
# artifacts = unpickle(artifacts)

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


class PredEndpointInputSchema(BaseModel):
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


# def predict(input_data: PredEndpointInputSchema) -> t.Dict:
#     """
#     API endpoint to get predictions for one single data point
#     """
#     s3 = boto3.client("s3")

#     inputs = {
#         "customerID": input_data.customerID,
#         "gender": input_data.gender,
#         "SeniorCitizen": input_data.SeniorCitizen,
#         "Partner": input_data.Partner,
#         "Dependents": input_data.Dependents,
#         "tenure": input_data.tenure,
#         "PhoneService": input_data.PhoneService,
#         "MultipleLines": input_data.MultipleLines,
#         "InternetService": input_data.InternetService,
#         "OnlineSecurity": input_data.OnlineSecurity,
#         "OnlineBackup": input_data.OnlineBackup,
#         "DeviceProtection": input_data.DeviceProtection,
#         "TechSupport": input_data.TechSupport,
#         "StreamingTV": input_data.StreamingTV,
#         "StreamingMovies": input_data.StreamingMovies,
#         "Contract": input_data.Contract,
#         "PaperlessBilling": input_data.PaperlessBilling,
#         "PaymentMethod": input_data.PaymentMethod,
#         "MonthlyCharges": input_data.MonthlyCharges,
#         "TotalCharges": input_data.TotalCharges,
#     }
#     input_df = pd.DataFrame(inputs, index=[0])
#     file_name = f"{str(uuid.uuid4())}.json"
#     response = {}
#     response["errors"] = {}
#     try:
#         INPUT_SCHEMA.validate(input_df)
#         input_df["TotalCharges"] = input_df["TotalCharges"].replace(
#             to_replace=" ", value="0"
#         )
#         input_ohe = input_df[config.data.get("CAT_COLS_OHE")]
#         input_ohe_trans = artifacts["encoder_ohe"].transform(input_ohe)
#         input_ohe_trans__df = pd.DataFrame(
#             input_ohe_trans.toarray(),
#             columns=artifacts["encoder_ohe"].get_feature_names_out(),
#         )
#         input_oe = input_df[config.data.get("CAT_COLS_OE")]
#         input_oe_trans: pd.DataFrame = artifacts["encoder_oe"].transform(input_oe)
#         input_scale = input_df[config.data.get("NUM_COLS")]
#         input_scale_trans: pd.DataFrame = artifacts["scaler_standard"].transform(
#             input_scale
#         )
#         input_to_predict = pd.concat(
#             [
#                 input_ohe_trans__df.reset_index(drop=True),
#                 input_oe_trans.reset_index(drop=True),
#                 input_scale_trans.reset_index(drop=True),
#             ],
#             axis=1,
#         )
#         prediction = artifacts["model"].predict(input_to_predict)
#         prediction_proba = artifacts["model"].predict_proba(input_to_predict)
#         response["prediction"] = prediction
#         response["prediction_proba"] = prediction_proba
#         response["inputs"] = inputs
#     except pa.errors.SchemaError as err:
#         response["errors"]["schema_failure_cases"] = err.failure_cases
#         response["errors"]["data"] = err.data


def main():
    encoder_oe: preprocessing.OrdinalEncoder = pickle.load(
        open("./models/artifacts/encoder_oe_.pkl", "rb")
    )
    encoder_ohe: preprocessing.OneHotEncoder = pickle.load(
        open("./models/artifacts/encoder_ohe_.pkl", "rb")
    )
    scaler_stand: preprocessing.StandardScaler = pickle.load(
        open("./models/artifacts/scaler_standard_.pkl", "rb")
    )

    model: t.Union[ensemble.RandomForestClassifier, xgb.XGBClassifier] = pickle.load(
        open("./models/XGBoost_best_.pkl", "rb")
    )

    payload = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
    }

    df: pd.DataFrame = pd.DataFrame(payload, index=[0])
    try:
        INPUT_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)
        print("\nDataFrame object that failed validation:")
        print(err.data)
        raise Exception("Schema errors and failure cases")

    exit()

    CAT_COLS_OE = [
        "OnlineSecurity",
        "MultipleLines",
        "Dependents",
        "DeviceProtection",
        "StreamingTV",
        "OnlineBackup",
        "gender",
        "SeniorCitizen",
        "PhoneService",
        "Partner",
        "PaperlessBilling",
        "StreamingMovies",
        "TechSupport",
    ]
    X_oe__test = df[CAT_COLS_OE].values
    X_oe_trans__test = encoder_oe.transform(X_oe__test)
    X_oe_trans__test = pd.DataFrame(X_oe_trans__test, columns=CAT_COLS_OE)

    CAT_COLS_OHE = ["PaymentMethod", "Contract", "InternetService"]
    X_ohe__test = df[CAT_COLS_OHE].values
    X_ohe_trans__test = encoder_ohe.transform(X_ohe__test)
    X_ohe_trans__test = pd.DataFrame(
        X_ohe_trans__test.toarray(), columns=encoder_ohe.get_feature_names_out()
    )

    NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_scaled__test = df[NUM_COLS].values
    X_scaled_trans__test = scaler_stand.transform(X_scaled__test)
    X_scaled_trans__test = pd.DataFrame(X_scaled_trans__test, columns=NUM_COLS)

    input_to_predict: pd.DataFrame = pd.concat(
        [
            X_ohe_trans__test.reset_index(drop=True),
            X_oe_trans__test.reset_index(drop=True),
            X_scaled_trans__test.reset_index(drop=True),
        ],
        axis=1,
    )

    model_prediction_proba = (
        model.predict_proba(input_to_predict.values).squeeze().tolist()
    )
    model_prediction = model.predict(input_to_predict.values).squeeze()
    print(model_prediction_proba)
    print(model_prediction)
    # _ = predict(PredEndpointInputSchema(**payload))


if __name__ == "__main__":
    main()
