import pandera as pa
from pathlib import Path
import pandas as pd

from schemas import TRAINING_SCHEMA

try:
    df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    TRAINING_SCHEMA.validate(df, lazy=True)
except pa.errors.SchemaErrors as err:
    print("Schema errors and failure cases:")
    print(err.failure_cases)
    print("\nDataFrame object that failed validation:")
    print(err.data)
