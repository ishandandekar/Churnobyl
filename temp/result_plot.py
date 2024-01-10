import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import random

results = dict()
random.seed(69)
results["lr"] = [round(random.uniform(0, 1), 2) for _ in range(8)]
results["svm"] = [round(random.uniform(0, 1), 2) for _ in range(8)]
results["knn"] = [round(random.uniform(0, 1), 2) for _ in range(8)]
results["voting"] = [round(random.uniform(0, 1), 2) for _ in range(8)]

models = list(results.keys())
data = pl.DataFrame(results).transpose()
data.columns = [
    "train_acc",
    "train_prec",
    "train_rec",
    "train_f1",
    "test_acc",
    "test_prec",
    "test_rec",
    "test_f1",
]
data = data.to_dict(as_series=False)
print(data)
print(models)


def grouped_bar_plot(data: dict, models: list):
    fig, ax = plt.subplots(layout="constrained", figsize=(12, 7))
    x = np.arange(len(models))
    width = 0.1
    multiplier = 0
    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ybound(0, 1.00001)
    ax.set_ylabel("Metrics")
    ax.set_title("Training results")
    ax.set_xlabel("Models")
    ax.set_xticks(x + width, models)
    ax.legend(loc="upper left")
    plt.show()


grouped_bar_plot(data=data, models=models)
