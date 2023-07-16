from pathlib import Path

import pandas as pd
import pandera as pa

data_schemas_path = Path.cwd() / "data_schemas"
assert data_schemas_path.is_dir() == True

df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
training_schema = pa.infer_schema(df)
churn_schema = pa.infer_schema(df[["Churn"]])
input_schema = pa.infer_schema(df.drop(columns=["Churn"]))

with open(data_schemas_path / "TRAINING_SCHEMA.py", "w") as file:
    file.write(training_schema.to_script())
with open(data_schemas_path / "CHURN_SCHEMA.py", "w") as file:
    file.write(churn_schema.to_script())
with open(data_schemas_path / "INPUT_SCHEMA.py", "w") as file:
    file.write(input_schema.to_script())
