import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from mlflow.models import infer_signature
import os
import mlflow
from urllib.parse import urlparse

# Loading environment variables
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/aljaziz/Machine-Learning-Pipeline-with-DVC-and-MLflow.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = "aljaziz"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "b6ed71240834547f3960998600a660253ef63a72"

# Load parameters
params = yaml.safe_load(open("params.yaml"))["train"]


def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search


def train(data_path, model_path, random_state, n_estiamtors=None, max_depth=None):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(
        "https://dagshub.com/aljaziz/Machine-Learning-Pipeline-with-DVC-and-MLflow.mlflow"
    )

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state, test_size=0.2
        )
        signature = infer_signature(X_train, y_train)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param(
            "best_min_samples_split", grid_search.best_params_["min_samples_split"]
        )
        mlflow.log_param(
            "best_min_samples_leaf", grid_search.best_params_["min_samples_leaf"]
        )

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # mlflow.sklearn.log_model(best_model, "model", signature=signature)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        filename = model_path
        pickle.dump(best_model, open(filename, "wb"))
        mlflow.log_artifacts("models", artifact_path="models")
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(params["data"], params["model"], params["random_state"])
