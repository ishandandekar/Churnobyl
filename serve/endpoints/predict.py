from typing import Union

import pandas as pd
import sklearn
from pydantic import BaseModel, Field


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

    class Config:
        from_attributes = True


def predict(
    data: dict,
    model_pipeline: sklearn.pipeline.Pipeline,
    label_encoder: sklearn.preprocessing.LabelBinarizer,
) -> dict:
    da = {k: [v] for k, v in data.items()}
    df = pd.DataFrame(da)
    pred_label = model_pipeline.predict(df)
    pred_class = label_encoder.inverse_transform(pred_label)
    pred_prob = model_pipeline.predict_proba(df)

    return {"prediction_label": pred_class, "prediction_probability": pred_prob}
