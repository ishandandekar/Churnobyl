import pandas as pd

df = pd.read_csv(
    "/home/ishan/Desktop/programs/pythonfiles/churninator/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
file_length = round(df.shape[0] * 0.5)
partition_1 = df[:file_length]
partition_2 = df[file_length:]
partition_1.to_csv("partition_1.csv", index=False)
partition_2.to_csv("partition_2.csv", index=False)
