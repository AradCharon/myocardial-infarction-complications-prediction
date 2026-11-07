
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "selected" / "X_test.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("CONSTANT FEATURE ANALYSIS")
    print("=" * 60)

    X_train = pd.read_csv(TRAIN_PATH)
    X_test = pd.read_csv(TEST_PATH)

    constant_features = [
        col for col in X_train.columns
        if X_train[col].nunique(dropna=True) <= 1
    ]

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    print("\nCONSTANT FEATURES IN TRAIN")

    if not constant_features:
        print("No constant features found.")
    else:
        for col in constant_features:
            train_values = X_train[col].dropna().unique().tolist()
            test_values = X_test[col].dropna().unique().tolist()

            print(f"\nFeature: {col}")
            print(f"Train unique values: {train_values}")
            print(f"Test unique values:  {test_values}")

    report = pd.DataFrame({
        "feature": constant_features,
        "train_unique_count": [
            X_train[col].nunique(dropna=True)
            for col in constant_features
        ],
        "train_values": [
            str(X_train[col].dropna().unique().tolist())
            for col in constant_features
        ],
        "test_unique_count": [
            X_test[col].nunique(dropna=True)
            for col in constant_features
        ],
        "test_values": [
            str(X_test[col].dropna().unique().tolist())
            for col in constant_features
        ],
    })

    output_path = OUTPUT_DIR / "constant_feature_analysis.csv"
    report.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"Number of constant features: {len(constant_features)}")
    print(f"Report saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
