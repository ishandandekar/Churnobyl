import pandas as pd
import pandera as pa
from data_schema import input_schema, churn_schema

with open("input_schema.yaml", "w") as file:
    file.write(input_schema.to_yaml())

with open("churn_schema.yaml", "w") as file:
    file.write(churn_schema.to_yaml())
