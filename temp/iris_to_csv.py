from sklearn import datasets, model_selection
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names).sample(n=1)
df.to_csv("iris_csv_for_tests.csv", index=False)
