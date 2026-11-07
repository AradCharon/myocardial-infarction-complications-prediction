
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/MI.data")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

df = pd.read_csv(
    DATA_PATH,
    header=None,
    sep=",",
    na_values=["?"]
)

print("=" * 65)
print("1. DATASET DIMENSIONS")
print("=" * 65)

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nExpected dimensions from dataset documentation:")
print("Approximately 1700 rows and 124 columns (1 ID + 111 features + 12 targets).")

print("\n" + "=" * 65)
print("2. FIRST 5 OBSERVATIONS")
print("=" * 65)

print(df.head().to_string())

print("\n" + "=" * 65)
print("3. DATA TYPES")
print("=" * 65)

print(df.dtypes.value_counts())
print("\nDetailed information:")
df.info()

print("\n" + "=" * 65)
print("4. MISSING VALUES")
print("=" * 65)

missing_count = df.isna().sum()
missing_percent = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent
})

missing_summary = missing_summary.sort_values(
    by="missing_count",
    ascending=False
)

print(f"Total missing values: {df.isna().sum().sum()}")
print(f"Overall missing percentage: {df.isna().sum().sum() / df.size * 100:.2f}%")

print("\nColumns with the most missing values:")
print(missing_summary[missing_summary["missing_count"] > 0].head(20))

print("\n" + "=" * 65)
print("5. DUPLICATE CHECK")
print("=" * 65)

print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate IDs: {df.iloc[:, 0].duplicated().sum()}")

print("\n" + "=" * 65)
print("6. UNIQUE VALUES PER COLUMN")
print("=" * 65)

for col in df.columns:
    print(
        f"Column {col:>3}: "
        f"{df[col].nunique(dropna=True)} unique values"
    )
