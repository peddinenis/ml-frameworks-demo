from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_data():
    """
    Load the train and test splits created earlier.
    """
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def load_preprocessor():
    """
    Load the fitted scikit-learn preprocessing pipeline.

    We reuse the exact same preprocessing for TensorFlow so all
    frameworks are compared on the same input features.
    """
    return joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")


def prepare_features(train_df, test_df, preprocessor):
    """
    Transform raw tabular data into numeric matrices.

    TensorFlow expects dense numeric arrays, so sparse outputs are
    converted to dense arrays when needed.
    """
    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    y_train = (train_df["target"] == ">50K").astype(int).values
    y_test = (test_df["target"] == ">50K").astype(int).values

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    return X_train_processed, X_test_processed, y_train, y_test


def build_model(input_dim: int):
    """
    Build a simple feed-forward neural network for binary classification.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    """
    End-to-end TensorFlow training workflow:
    1. Load data
    2. Load fitted preprocessor
    3. Transform features into numeric arrays
    4. Train a neural network
    5. Evaluate on the test set
    6. Save the trained model
    """
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    train_df, test_df = load_data()
    preprocessor = load_preprocessor()

    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df, preprocessor)

    # Handle class imbalance so the minority class is learned better.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model = build_model(X_train.shape[1])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=1,
    )

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

    model.save(ARTIFACTS_DIR / "tensorflow_model.keras")
    print("\nSaved model to artifacts/tensorflow_model.keras")


if __name__ == "__main__":
    main()