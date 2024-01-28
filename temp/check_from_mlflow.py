from pprint import pprint

import mlflow

runs = mlflow.search_runs(["1"], order_by=["end_time"]).to_dict()
pprint(runs)
# print(runs)
# run_id = runs.iloc[-1, :]["run_id"]
# with mlflow.start_run(run_id=run_id):
#     m = mlflow.sklearn.load_model(f"s3://mlflow-churnobyl/1/{run_id}/artifacts/model")
#     print(type(m))
#     print(m)
