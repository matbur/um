from pathlib import Path
import re
import io
import json

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/{0}/{0}.data"
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
DATASETS = (
    # name, label-column, categorical, ignore
    ("balance-scale", 4, (0,), ()),
    ("breast-cancer-wisconsin", 10, (), (0,)),
    ("ecoli", 8, (8,), (0,)),
    ("glass", 10, (), (0,)),
    ("haberman", 3, (), ()),
    ("iris", 4, (4,), ()),
    ("letter-recognition", 0, (0,), ()),
    ("wine", 0, (), ()),
)


def load_data(dataset: str, label_col: int, categorical=(), ignore=()) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    data_file = DATA_DIR / f"{dataset}.csv"

    if data_file.exists():
        df = pd.read_csv(data_file.as_posix(), header=None)

        for i in ignore:
            df = df.drop(i, axis=1)

        X = df.copy()
        y = X.pop(label_col)
        return X, y

    print(f"Downloading {dataset}")

    resp = requests.get(URL.format(dataset))
    resp.raise_for_status()

    text = re.sub(" +", ",", resp.text)
    buf = io.StringIO(text)
    df = pd.read_csv(buf, header=None)

    for i in categorical:
        df[i] = LabelEncoder().fit(df[i]).transform(df[i])

    if dataset == "breast-cancer-wisconsin":
        df = df[df[6] != "?"]
        df[6] = df[6].astype(int)

    df.to_csv(data_file.as_posix(), index=False, header=None)

    for i in ignore:
        df = df.drop(i, axis=1)

    X = df.copy()
    y = X.pop(label_col)
    return X, y


def get_score(
    model: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> int:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score


def plot(dataset, scores):
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_file = RESULTS_DIR / f"{dataset}.png"

    bottom = min(scores.values()) ** 1.5
    heights = [i - bottom for i in scores.values()]

    fig, ax = plt.subplots(1, 1)
    ax.bar(scores.keys(), heights, bottom=bottom, color="black")
    ax.grid()
    ax.set_title(dataset)
    fig.savefig(plot_file.as_posix())


def main():
    np.random.seed(42)
    results = {}
    for i in DATASETS:
        dataset = i[0]
        print(dataset)
        X, y = load_data(*i)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        scores = {
            "bagging": get_score(BaggingClassifier(), X_train, y_train, X_test, y_test),
            "adaboost": get_score(
                AdaBoostClassifier(), X_train, y_train, X_test, y_test
            ),
            "bagging_adaboost": get_score(
                BaggingClassifier(AdaBoostClassifier()),
                X_train,
                y_train,
                X_test,
                y_test,
            ),
        }

        for method, score in scores.items():
            print(method, score)

        plot(dataset, scores)
        results[dataset] = scores

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
