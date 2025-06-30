import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/aljaziz/Machine-Learning-Pipeline-with-DVC-and-MLflow.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = "aljaziz"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "b6ed71240834547f3960998600a660253ef63a72"

# Load parameters
params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns="Outcome")
    y = data["Outcome"]

    mlflow.set_tracking_uri(
        "https://dagshub.com/aljaziz/Machine-Learning-Pipeline-with-DVC-and-MLflow.mlflow"
    )
    model = pickle.load(open(model_path, "rb"))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy {accuracy}")


if __name__ == "__main__":
    evaluate(params["data"], params["model"])
