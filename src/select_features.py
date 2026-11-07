
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "X_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "X_test.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "selected"
REPORT_DIR = ROOT / "outputs" / "metrics"

MISSING_THRESHOLD = 0.50


def main():
    print("=" * 60)
    print("FEATURE SELECTION - MISSING VALUES")
    print("=" * 60)

    X_train = pd.read_csv(TRAIN_PATH)
    X_test = pd.read_csv(TEST_PATH)

    print(f"\nTrain shape before: {X_train.shape}")
    print(f"Test shape before:  {X_test.shape}")

    # Calculate missing percentages using Train only
    missing_ratio = X_train.isnull().mean()

    # Select features with more than 50% missing values
    dropped_features = missing_ratio[
        missing_ratio > MISSING_THRESHOLD
    ].index.tolist()

    selected_features = [
        col for col in X_train.columns
        if col not in dropped_features
    ]

    # Apply the same feature selection to Train and Test
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Save selected datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_selected.to_csv(
        OUTPUT_DIR / "X_train.csv", index=False
    )

    X_test_selected.to_csv(
        OUTPUT_DIR / "X_test.csv", index=False
    )

    # Save feature selection report
    report = pd.DataFrame({
        "feature": missing_ratio.index,
        "missing_ratio": missing_ratio.values,
        "missing_percent": missing_ratio.values * 100,
        "selected": [
            col not in dropped_features
            for col in missing_ratio.index
        ]
    })

    report.to_csv(
        REPORT_DIR / "missingness_feature_selection.csv",
        index=False
    )

    print("\nDROPPED FEATURES")
    for feature in dropped_features:
        print(
            f"{feature}: "
            f"{missing_ratio[feature] * 100:.2f}%"
        )

    print(f"\nNumber of dropped features: {len(dropped_features)}")
    print(f"Number of selected features: {len(selected_features)}")

    print(f"\nTrain shape after: {X_train_selected.shape}")
    print(f"Test shape after:  {X_test_selected.shape}")

    print("\nOutput files saved successfully.")


if __name__ == "__main__":
    main()
