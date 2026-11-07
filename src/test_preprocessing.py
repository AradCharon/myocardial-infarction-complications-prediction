
from pathlib import Path

import pandas as pd

from preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT / "data" / "processed" / "selected" / "X_train.csv"
)

X_train = pd.read_csv(TRAIN_PATH)

preprocessor = build_preprocessor(scale_continuous=True)

X_train_transformed = preprocessor.fit_transform(X_train)

print("=" * 60)
print("PREPROCESSING TEST")
print("=" * 60)

print("Original shape:", X_train.shape)
print("Transformed shape:", X_train_transformed.shape)

print(
    "Removed constant features:",
    X_train.shape[1] - X_train_transformed.shape[1]
)

print(
    "Missing values after preprocessing:",
    pd.isna(X_train_transformed).sum()
)

print("=" * 60)
