import warnings
import os
import typing as t
import uuid
import json
from http import HTTPStatus
import fastapi
import wandb
import boto3
from pathlib import Path
import pandas as pd
import pandera as pa
import uvicorn
import utils
import endpoints


warnings.simplefilter("ignore")


# FIXME: Add this func to both endpoints
# def validate_input(input: pd.DataFrame, schema: pa.DataFrameSchema):
#     """Validate input against schema"""
#     try:
#         schema.validate(input)
#     except pa.errors.SchemaErrors as e:
#         raise Exception(e)


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
    # target_encoder_path = Path(preprocessor_path) / "target_encoder_.pkl"
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
        # "target_encoder": target_encoder_path,
    }
    return artifacts


CONFIG_PATH = Path("./serve/serve-config.yaml")
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
config = utils.set_config(config_path=CONFIG_PATH, WANDB_API_KEY=WANDB_API_KEY)
artifacts = load_artifacts()
artifacts = utils.unpickle(artifacts)

# TODO: Add code for FastAPI and then dockerize
app = fastapi.FastAPI(
    title="Churnobyl", description="Predict the churn, and DO NOT BURN", version="0.1"
)


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
def predict(data: utils.PredEndpointInputSchema) -> t.Dict:
    """
    API endpoint to get predictions for one single data point
    """
    return endpoints.predict(data.model_dump())
    s3 = boto3.client("s3")

    # inputs = {
    #     "customerID": input_data.customerID,
    #     "gender": input_data.gender,
    #     "SeniorCitizen": input_data.SeniorCitizen,
    #     "Partner": input_data.Partner,
    #     "Dependents": input_data.Dependents,
    #     "tenure": input_data.tenure,
    #     "PhoneService": input_data.PhoneService,
    #     "MultipleLines": input_data.MultipleLines,
    #     "InternetService": input_data.InternetService,
    #     "OnlineSecurity": input_data.OnlineSecurity,
    #     "OnlineBackup": input_data.OnlineBackup,
    #     "DeviceProtection": input_data.DeviceProtection,
    #     "TechSupport": input_data.TechSupport,
    #     "StreamingTV": input_data.StreamingTV,
    #     "StreamingMovies": input_data.StreamingMovies,
    #     "Contract": input_data.Contract,
    #     "PaperlessBilling": input_data.PaperlessBilling,
    #     "PaymentMethod": input_data.PaymentMethod,
    #     "MonthlyCharges": input_data.MonthlyCharges,
    #     "TotalCharges": input_data.TotalCharges,
    # }
    inputs = input_data.dict()
    input_df = pd.DataFrame(inputs, index=[0])
    file_name = f"{str(uuid.uuid4())}.json"
    response = {}
    response["errors"] = {}
    try:
        utils.INPUT_SCHEMA.validate(input_df, lazy=True)
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
        response["inputs"] = inputs
        json_string = json.dumps(response)
        s3_response = s3.put_object(
            Body=json_string,
            Bucket=config.data.get("BUCKET_NAME"),
            Key="api_logs/predict_logs/" + file_name,
        )
        if s3_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            response["message"] = "Prediction logged successfully"
            response["status-code"] = HTTPStatus.OK
            response["data"] = {"file_name": file_name}
        else:
            response["message"] = "Prediction logging failed"
            response["status-code"] = HTTPStatus.INTERNAL_SERVER_ERROR
            response["data"] = {}

    except pa.errors.SchemaError as err:
        response["errors"]["schema_failure_cases"] = err.failure_cases
        response["errors"]["data"] = err.data


# TODO: and this
@app.post("/flag", tags=["Flagging"])
def flag(data: utils.FlagEndpointInputSchema):
    """
    API endpoint to flag a prediction. Must contain the predicted label, prediction probability and the actual label
    """
    return endpoints.flag(data.model_dump())
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
        utils.FLAG_SCHEMA.validate(flag_df)
        s3 = boto3.client("s3")

        # Generate a unique file name
        file_name = f"{str(uuid.uuid4())}.json"
        json_string = json.dumps(flag_data)
        try:
            s3_response = s3.put_object(
                Body=json_string,
                Bucket=config.s3.get("BUCKET_NAME"),
                Key="api_logs/flag_logs/" + file_name,
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


if __name__ == "__main__":
    uvicorn.run(app=app, port=8000, host="0.0.0.0")
