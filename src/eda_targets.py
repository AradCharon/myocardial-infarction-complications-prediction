
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/MI.data")

df = pd.read_csv(
    DATA_PATH,
    header=None,
    sep=",",
    na_values=["?"]
)

target_names = [
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

targets = df.iloc[:, 112:124].copy()
targets.columns = target_names

print("=" * 65)
print("TARGET VARIABLES")
print("=" * 65)

print(f"Target shape: {targets.shape}")

for column in target_names:
    print(f"\n{column}")
    print(targets[column].value_counts(dropna=False).sort_index())
    print("\nPercentage:")
    print(
        (targets[column].value_counts(normalize=True, dropna=False)
         .sort_index() * 100).round(2)
    )

print("\n" + "=" * 65)
print("REQUIRED TARGETS")
print("=" * 65)

required_targets = ["LET_IS", "ZSN", "FIBR_PREDS"]

for column in required_targets:
    print(f"\n{column}")
    print(targets[column].value_counts().sort_index())

print("\n" + "=" * 65)
print("LET_IS BINARY TRANSFORMATION")
print("=" * 65)

let_is_binary = (targets["LET_IS"] > 0).astype(int)

print(let_is_binary.value_counts().sort_index())
print("\nPercentages:")
print((let_is_binary.value_counts(normalize=True).sort_index() * 100).round(2))

print("\n" + "=" * 65)
print("TARGET MISSING VALUES")
print("=" * 65)

print(targets.isna().sum())

print("\n" + "=" * 65)
print("TARGET VALUE VALIDATION")
print("=" * 65)

for column in target_names[:-1]:
    invalid = sorted(set(targets[column].dropna().unique()) - {0, 1})
    print(f"{column}: invalid values = {invalid}")

invalid_let_is = sorted(
    set(targets["LET_IS"].dropna().unique()) - set(range(8))
)

print(f"LET_IS: invalid values = {invalid_let_is}")
