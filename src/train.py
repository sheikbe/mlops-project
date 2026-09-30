import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
EXPERIMENT_NAME = "iris-classifier"
REGISTERED_MODEL_NAME = "iris-classifier"


def train(c: float = 1.0, max_iter: int = 200, model_path: str = MODEL_PATH) -> float:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = LogisticRegression(C=c, max_iter=max_iter).fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        mlflow.log_params({"model": "LogisticRegression", "C": c, "max_iter": max_iter})
        mlflow.log_metrics(
            {"accuracy": accuracy, "f1_macro": f1_score(y_test, preds, average="macro")}
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_test[:2],
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    joblib.dump(model, model_path)
    print(f"accuracy={accuracy:.3f} saved model to {model_path}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Iris classifier and log it to MLflow")
    parser.add_argument("--C", type=float, default=1.0, dest="c")
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)
