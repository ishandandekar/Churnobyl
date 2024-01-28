import os
import pickle

import mlflow
from rich.pretty import pprint
from sklearn import compose, pipeline

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name="churnobyl")
preprocessor_path = "churninator/models/artifacts/feature_transformer.pkl"
preprocessor: compose.ColumnTransformer = pickle.loads(
    open(preprocessor_path, "rb").read()
)
model_path = "churninator/models/lr.pkl"
model = pickle.loads(open(model_path, "rb").read())
pipe = pipeline.Pipeline(steps=[("column_transformer", preprocessor), ("model", model)])
pprint(pipe)
with mlflow.start_run():
    mlflow.sklearn.log_model(pipe, artifact_path="model")
