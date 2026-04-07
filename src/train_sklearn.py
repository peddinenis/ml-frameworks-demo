from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Configuration constants
# -------------------------------
RANDOM_STATE = 42
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_data():
    """
    Load pre-split training and testing datasets from CSV files.

    Returns:
        tuple: (train_df, test_df) as pandas DataFrames
    """
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def build_preprocessor(X: pd.DataFrame):
    """
    Build a preprocessing pipeline that:
    - Imputes missing values
    - Scales numeric features
    - One-hot encodes categorical features

    Args:
        X (pd.DataFrame): Feature dataset

    Returns:
        ColumnTransformer: Preprocessing pipeline
    """

    # Identify feature types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Pipeline for numeric features:
    # - Fill missing values with median
    # - Standardize values (mean=0, std=1)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline for categorical features:
    # - Fill missing values with most frequent value
    # - Convert categories to one-hot encoded vectors
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine both pipelines into one transformer
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def main():
    """
    Main training workflow:
    1. Load data
    2. Split features and target
    3. Build preprocessing + model pipeline
    4. Train model
    5. Evaluate performance
    6. Save trained model
    """

    # Load datasets
    train_df, test_df = load_data()

    # Separate features and target variable
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # Create full pipeline:
    # preprocessing → model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save trained pipeline (preprocessing + model together)
    joblib.dump(model, ARTIFACTS_DIR / "sklearn_model.joblib")
    print("\nSaved model to artifacts/sklearn_model.joblib")


if __name__ == "__main__":
    main()