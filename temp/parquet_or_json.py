import os

import polars as pl

data = [{"hi": "hello", "age": 21, "gender": "prefer not to say"}]

df = pl.from_dicts(data=data)
df.write_parquet("parquet_data.parquet")
df.write_json("json_data.json", pretty=True, row_oriented=True)
p_mem = os.stat("parquet_data.parquet").st_size
j_mem = os.stat("json_data.json").st_size
print(p_mem, j_mem)

if p_mem > j_mem:
    print("JSON wins")
else:
    print("PARQUET wins")
