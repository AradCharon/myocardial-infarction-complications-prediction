
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

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
ids = df["ID"].copy()

y["LET_IS"] = (y["LET_IS"] > 0).astype(int)

indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y["LET_IS"]
)

X_train = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)

y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)

id_train = ids.iloc[train_indices].reset_index(drop=True)
id_test = ids.iloc[test_indices].reset_index(drop=True)

split_dir = root / "data" / "processed"
split_dir.mkdir(parents=True, exist_ok=True)

X_train.to_csv(split_dir / "X_train.csv", index=False)
X_test.to_csv(split_dir / "X_test.csv", index=False)

y_train.to_csv(split_dir / "y_train.csv", index=False)
y_test.to_csv(split_dir / "y_test.csv", index=False)

id_train.to_csv(split_dir / "id_train.csv", index=False)
id_test.to_csv(split_dir / "id_test.csv", index=False)

train_indices_df = pd.DataFrame({"original_index": train_indices})
test_indices_df = pd.DataFrame({"original_index": test_indices})

train_indices_df.to_csv(split_dir / "train_indices.csv", index=False)
test_indices_df.to_csv(split_dir / "test_indices.csv", index=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("DATA SPLIT")
print("Total samples:", len(df))
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
print("Number of features:", X_train.shape[1])

print("\nLET_IS DISTRIBUTION")
print("Full dataset:")
print(y["LET_IS"].value_counts().sort_index())
print(y["LET_IS"].value_counts(normalize=True).sort_index())

print("\nTrain:")
print(y_train["LET_IS"].value_counts().sort_index())
print(y_train["LET_IS"].value_counts(normalize=True).sort_index())

print("\nTest:")
print(y_test["LET_IS"].value_counts().sort_index())
print(y_test["LET_IS"].value_counts(normalize=True).sort_index())

print("\nCROSS VALIDATION")
print("Number of folds:", cv.get_n_splits())
print("Random state:", 42)

print("\nSaved train/test datasets and original indices.")
