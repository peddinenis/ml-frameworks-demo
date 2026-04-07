from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")


class ChurnMLP(nn.Module):
    """PyTorch model architecture used during training."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def load_data():
    """Load the saved train/test split."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def get_xy(df):
    """Split features and binary target."""
    X = df.drop(columns=["target"])
    y = (df["target"] == ">50K").astype(int).values
    return X, y


def evaluate_binary(y_true, y_pred, y_prob):
    """Compute standard binary classification metrics."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def evaluate_sklearn():
    """Evaluate the saved scikit-learn pipeline."""
    _, test_df = load_data()
    X_test, y_test = get_xy(test_df)

    model = joblib.load(ARTIFACTS_DIR / "sklearn_model.joblib")

    y_pred_raw = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Convert string labels to binary integers so metrics match y_test
    y_pred = (y_pred_raw == ">50K").astype(int)

    return evaluate_binary(y_test, y_pred, y_prob)


def evaluate_tensorflow():
    """Evaluate the saved TensorFlow model."""
    _, test_df = load_data()
    X_test, y_test = get_xy(test_df)

    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    model = tf.keras.models.load_model(ARTIFACTS_DIR / "tensorflow_model.keras")
    y_prob = model.predict(X_test_processed, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    return evaluate_binary(y_test, y_pred, y_prob)


def evaluate_pytorch():
    """Evaluate the saved PyTorch model."""
    _, test_df = load_data()
    X_test, y_test = get_xy(test_df)

    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    model = ChurnMLP(input_dim=X_test_processed.shape[1])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "pytorch_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_test_tensor)).numpy()

    y_pred = (y_prob >= 0.5).astype(int)

    return evaluate_binary(y_test, y_pred, y_prob)


def main():
    """Run all model evaluations and print a comparison table."""
    results = {
        "scikit-learn": evaluate_sklearn(),
        "TensorFlow": evaluate_tensorflow(),
        "PyTorch": evaluate_pytorch(),
    }

    comparison = pd.DataFrame(results).T
    print("\nModel Comparison:\n")
    print(comparison)

    with open(ARTIFACTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved comparison to artifacts/model_comparison.json")


if __name__ == "__main__":
    main()