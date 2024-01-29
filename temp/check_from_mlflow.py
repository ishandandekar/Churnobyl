from pprint import pprint

import mlflow

runs = mlflow.search_runs(["1"], order_by=["end_time"])
pprint(runs)
run_id = runs.iloc[-1, :]["run_id"]
print(run_id)
with mlflow.start_run(run_id=run_id):
    m = mlflow.sklearn.load_model(f"s3://mlflow-churnobyl/1/{run_id}/artifacts/model")
    print(type(m))
    print(m)
