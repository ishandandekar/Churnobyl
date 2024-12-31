#
# import mlflow
#
# runs = mlflow.search_runs(["1"], order_by=["end_time"])
# pprint(runs)
# run_id = runs.iloc[-1, :]["run_id"]
# print(run_id)
# with mlflow.start_run(run_id=run_id):
#     m = mlflow.sklearn.load_model(f"s3://mlflow-churnobyl/1/{run_id}/artifacts/model")
#     print(type(m))
#     print(m)
#
import pickle

import mlflow
import pandas as pd
import sklearn

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

data = {k: [v] for k, v in payload.items()}

df = pd.DataFrame(data)

m: sklearn.pipeline.Pipeline = mlflow.sklearn.load_model("best_run_artifacts/model")

label_encoder = pickle.load(open("best_run_artifacts/label_binarizer.pkl", "rb"))
print(type(m))
print(m)
preds = m.predict(df)
pred_prob = m.predict_proba(df)
print(preds)
print(label_encoder.inverse_transform(preds))
