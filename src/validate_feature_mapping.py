
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = ROOT / "outputs" / "metrics" / "feature_type_profile.csv"

TRAIN_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "selected" / "X_test.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


VALID_TYPES = {"binary", "ordinal", "continuous"}


def main():
    print("=" * 65)
    print("FEATURE MAPPING VALIDATION")
    print("=" * 65)

    profile = pd.read_csv(PROFILE_PATH)
    X_train = pd.read_csv(TRAIN_PATH)
    X_test = pd.read_csv(TEST_PATH)

    # Basic validation
    required_columns = {"feature", "feature_type"}

    if not required_columns.issubset(profile.columns):
        raise ValueError(
            f"Profile must contain columns: {required_columns}"
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and Test columns do not match.")

    if profile["feature"].duplicated().any():
        raise ValueError("Duplicate feature names found in profile.")

    # Normalize feature types
    profile["feature_type"] = (
        profile["feature_type"].astype(str).str.strip().str.lower()
    )

    # Check invalid types
    invalid_types = set(profile["feature_type"]) - VALID_TYPES

    if invalid_types:
        raise ValueError(f"Invalid feature types: {invalid_types}")

    # Check feature coverage
    selected_features = set(X_train.columns)
    mapped_features = set(profile["feature"])

    missing_mapping = selected_features - mapped_features
    extra_mapping = mapped_features - selected_features

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Selected features: {len(selected_features)}")

    print("\nFEATURE MAPPING CHECK")

    if missing_mapping:
        print("Features without mapping:", sorted(missing_mapping))
    else:
        print("All selected features have a mapping.")

    if extra_mapping:
        print(
            "Mapped features not present in selected data:",
            sorted(extra_mapping)
        )

    if missing_mapping:
        raise ValueError("Some selected features have no type mapping.")

    # Keep only selected features
    selected_profile = (
        profile[profile["feature"].isin(selected_features)]
        .copy()
        .sort_values(["feature_type", "feature"])
        .reset_index(drop=True)
    )

    # Correct mapping must be verified explicitly
    if "IM_PG_P" in selected_profile["feature"].values:
        actual_type = selected_profile.loc[
            selected_profile["feature"] == "IM_PG_P",
            "feature_type"
        ].iloc[0]

        print(f"\nIM_PG_P mapped type: {actual_type}")

        if actual_type != "binary":
            print(
                "WARNING: IM_PG_P is not mapped as binary. "
                "Review its mapping before proceeding."
            )

    # Summary
    summary = (
        selected_profile["feature_type"]
        .value_counts()
        .reindex(["binary", "ordinal", "continuous"], fill_value=0)
    )

    print("\nFEATURE TYPE SUMMARY")
    print(summary.to_string())

    # Print feature names by type
    for feature_type in ["binary", "ordinal", "continuous"]:
        features = selected_profile.loc[
            selected_profile["feature_type"] == feature_type,
            "feature"
        ].tolist()

        print(f"\n{feature_type.upper()} FEATURES ({len(features)})")
        print(", ".join(features))

    # Check binary values in both datasets
    binary_features = selected_profile.loc[
        selected_profile["feature_type"] == "binary",
        "feature"
    ].tolist()

    invalid_binary = {}

    for col in binary_features:
        train_values = set(X_train[col].dropna().unique())
        test_values = set(X_test[col].dropna().unique())

        invalid_train = train_values - {0, 1}
        invalid_test = test_values - {0, 1}

        if invalid_train or invalid_test:
            invalid_binary[col] = {
                "train_invalid": sorted(invalid_train),
                "test_invalid": sorted(invalid_test),
            }

    print("\nBINARY VALIDATION")

    if invalid_binary:
        print("Invalid binary values found:")
        print(invalid_binary)
    else:
        print("All binary features contain only 0 and 1.")

    # Save selected mapping
    output_path = OUTPUT_DIR / "selected_feature_type_profile.csv"
    selected_profile.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print(f"Mapping report saved to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
