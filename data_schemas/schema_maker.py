import pandas as pd
import pandera as pa
from data_schema import CHURN_SCHEMA, INPUT_SCHEMA, TRAINING_SCHEMA

with open("training_schema.yaml", "w") as file:
    file.write(TRAINING_SCHEMA.to_yaml())

with open("input_schema.yaml", "w") as file:
    file.write(INPUT_SCHEMA.to_yaml())

with open("churn_schema.yaml", "w") as file:
    file.write(CHURN_SCHEMA.to_yaml())
