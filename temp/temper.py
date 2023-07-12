import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(iris.feature_names)
print(data.head(2))
sepals = data[["sepal length (cm)", "sepal width (cm)"]]
petals = data[["petal length (cm)", "petal width (cm)"]]
print(f"Sepal --> {sepals.shape}")
print(f"Petal --> {petals.shape}")
result = pd.concat([sepals, petals], axis=1)
print(f"Result --> {result.shape}")
print(f"Result columns --> {result.columns}")
