from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3


def load_data():
    """Load train and test CSV files."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def load_preprocessor():
    """Load the fitted scikit-learn preprocessing pipeline."""
    return joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")


def prepare_features(train_df, test_df, preprocessor):
    """
    Convert raw tabular data into dense numeric arrays for PyTorch.
    """
    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    y_train = (train_df["target"] == ">50K").astype(np.float32).values
    y_test = (test_df["target"] == ">50K").astype(np.float32).values

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    return X_train_processed, X_test_processed, y_train, y_test


class ChurnMLP(nn.Module):
    """Simple feed-forward neural network for binary classification."""

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


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    train_df, test_df = load_data()
    preprocessor = load_preprocessor()

    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df, preprocessor)

    # Class imbalance handling
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.astype(int),
    )
    pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ChurnMLP(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    # Simple train/validation split from the training set
    n_train = int(0.8 * len(train_dataset))
    n_val = len(train_dataset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_subset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_subset)
        print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ARTIFACTS_DIR / "pytorch_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "pytorch_model.pt"))
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).numpy()
        y_pred = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

    print("\nSaved model to artifacts/pytorch_model.pt")


if __name__ == "__main__":
    main()