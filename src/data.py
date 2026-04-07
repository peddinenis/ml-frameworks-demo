from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def load_adult_income():
    dataset = fetch_openml("adult", version=2, as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns={"class": "target"})
    df = df.replace("?", pd.NA)
    return df


def split_and_save(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path} ({train_df.shape})")
    print(f"Saved: {test_path} ({test_df.shape})")

    return train_df, test_df


def main():
    df = load_adult_income()
    print("Dataset loaded.")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nTarget distribution:")
    print(df["target"].value_counts(normalize=True).round(3))

    split_and_save(df)


if __name__ == "__main__":
    main()