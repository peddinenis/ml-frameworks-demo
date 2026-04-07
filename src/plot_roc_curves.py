from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)


class ChurnMLP(nn.Module):
    """
    PyTorch model architecture used during training.

    This must match the architecture used in training so that
    saved weights can be loaded correctly.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # Sequential feed-forward network
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
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input features

        Returns:
            torch.Tensor: Raw logits (before sigmoid)
        """
        return self.network(x).squeeze(1)


def load_test_data():
    """
    Load the test dataset and separate features and labels.

    Returns:
        X_test (pd.DataFrame): Input features
        y_test (np.ndarray): Binary target labels (0/1)
    """
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Split features and target
    X_test = test_df.drop(columns=["target"])

    # Convert target to binary (<=50K → 0, >50K → 1)
    y_test = (test_df["target"] == ">50K").astype(int).values

    return X_test, y_test


def preprocess_features(X_test):
    """
    Apply the shared preprocessing pipeline to the test data.

    Converts categorical and numeric features into a fully numeric
    representation suitable for ML models.

    Args:
        X_test (pd.DataFrame)

    Returns:
        np.ndarray: Processed feature matrix
    """
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")

    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrix to dense if needed (required for DL frameworks)
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    return X_test_processed


def get_sklearn_probs(X_test):
    """
    Get predicted probabilities from the scikit-learn model.

    Args:
        X_test (pd.DataFrame)

    Returns:
        np.ndarray: Probability of positive class (>50K)
    """
    model = joblib.load(ARTIFACTS_DIR / "sklearn_model.joblib")

    # Predict probability for class 1 (>50K)
    return model.predict_proba(X_test)[:, 1]


def get_tensorflow_probs(X_test_processed):
    """
    Get predicted probabilities from the TensorFlow model.

    Args:
        X_test_processed (np.ndarray)

    Returns:
        np.ndarray: Probability of positive class
    """
    model = tf.keras.models.load_model(ARTIFACTS_DIR / "tensorflow_model.keras")

    # Output is already sigmoid probability
    return model.predict(X_test_processed, verbose=0).ravel()


def get_pytorch_probs(X_test_processed):
    """
    Get predicted probabilities from the PyTorch model.

    Args:
        X_test_processed (np.ndarray)

    Returns:
        np.ndarray: Probability of positive class
    """
    # Rebuild model architecture and load trained weights
    model = ChurnMLP(input_dim=X_test_processed.shape[1])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "pytorch_model.pt", map_location="cpu"))
    model.eval()

    # Convert input to tensor
    X_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)

        # Apply sigmoid to convert logits → probabilities
        probs = torch.sigmoid(logits).numpy()

    return probs


def plot_roc_curve(y_test, prob_dict):
    """
    Plot ROC curves for all models on a single chart.

    Args:
        y_test (np.ndarray): True labels
        prob_dict (dict): Mapping of model name → predicted probabilities
    """
    plt.figure(figsize=(8, 6))

    for model_name, y_prob in prob_dict.items():
        # Compute ROC curve points
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Compute AUC score
        auc_score = roc_auc_score(y_test, y_prob)

        # Plot curve
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})")

    # Plot baseline (random guessing)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    # Labels and styling
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Save the plot
    output_path = OUTPUT_DIR / "roc_curve_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved ROC curve plot to {output_path}")


def main():
    """
    End-to-end workflow:
    1. Load test data
    2. Apply preprocessing
    3. Get predictions from all models
    4. Plot ROC curves
    """
    X_test, y_test = load_test_data()

    # Preprocess features for deep learning models
    X_test_processed = preprocess_features(X_test)

    # Collect probabilities from each model
    prob_dict = {
        "scikit-learn": get_sklearn_probs(X_test),
        "TensorFlow": get_tensorflow_probs(X_test_processed),
        "PyTorch": get_pytorch_probs(X_test_processed),
    }

    # Generate comparison plot
    plot_roc_curve(y_test, prob_dict)


if __name__ == "__main__":
    main()