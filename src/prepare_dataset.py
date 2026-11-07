
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

raw_path = root / "data" / "raw" / "MI.data"
mapping_path = root / "outputs" / "metrics" / "column_mapping.csv"

df = pd.read_csv(raw_path, header=None, na_values="?")

mapping = pd.read_csv(mapping_path)
df.columns = mapping["column_name"].tolist()

target_columns = [
    "FIBR_PREDS",
    "PREDS_TAH",
    "JELUD_TAH",
    "FIBR_JELUD",
    "A_V_BLOK",
    "OTEK_LANC",
    "RAZRIV",
    "DRESSLER",
    "ZSN",
    "REC_IM",
    "P_IM_STEN",
    "LET_IS"
]

forbidden_columns = [
    "R_AB_1_n",
    "R_AB_2_n",
    "R_AB_3_n",
    "NA_R_1_n",
    "NA_R_2_n",
    "NA_R_3_n",
    "NOT_NA_1_n",
    "NOT_NA_2_n",
    "NOT_NA_3_n"
]

X = df.drop(columns=["ID"] + target_columns + forbidden_columns)
y = df[target_columns].copy()

feature_profile = pd.DataFrame({
    "feature": X.columns,
    "dtype": X.dtypes.astype(str).values,
    "missing_count": X.isna().sum().values,
    "missing_percent": (X.isna().mean() * 100).values,
    "unique_values": X.nunique(dropna=True).values
})

feature_profile["binary_candidate"] = feature_profile["unique_values"] <= 2
feature_profile["categorical_candidate"] = feature_profile["unique_values"].between(3, 10)
feature_profile["continuous_candidate"] = feature_profile["unique_values"] > 10

processed_dir = root / "data" / "processed"
metrics_dir = root / "outputs" / "metrics"

processed_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

X.to_csv(processed_dir / "X_admission.csv", index=False)
y.to_csv(processed_dir / "y_targets.csv", index=False)
feature_profile.to_csv(metrics_dir / "feature_profile.csv", index=False)

print("DATASET PREPARATION")
print("Original shape:", df.shape)
print("Input features:", X.shape)
print("Targets:", y.shape)
print("Total missing values in X:", X.isna().sum().sum())

print("\nFEATURE PROFILE")
print(feature_profile.to_string(index=False))

print("\nSaved files:")
print("data/processed/X_admission.csv")
print("data/processed/y_targets.csv")
print("outputs/metrics/feature_profile.csv")
