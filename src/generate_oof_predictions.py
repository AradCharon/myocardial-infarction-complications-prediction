
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

X_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
Y_PATH = ROOT / "data" / "processed" / "y_train.csv"

OUTPUT_DIR = ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "oof_predictions.csv"

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]

RANDOM_STATE = 42
N_SPLITS = 5

X_train = pd.read_csv(X_PATH)
y_train = pd.read_csv(Y_PATH)

if len(X_train) != len(y_train):
    raise ValueError("X_train and y_train have different row counts.")

if "LET_IS" not in y_train.columns:
    raise ValueError("LET_IS target not found.")

if not set(y_train["LET_IS"].dropna().unique()).issubset(set(range(8))):
    raise ValueError("Unexpected LET_IS values.")

y_train["LET_IS"] = (y_train["LET_IS"] > 0).astype(int)

for target in TARGETS:
    if target not in y_train.columns:
        raise ValueError(f"Target not found: {target}")

    if y_train[target].isna().any():
        raise ValueError(f"Missing values in target: {target}")

    if not set(y_train[target].unique()).issubset({0, 1}):
        raise ValueError(f"Target is not binary: {target}")

    if y_train[target].value_counts().min() < N_SPLITS:
        raise ValueError(f"Insufficient samples for CV: {target}")

models = {
    "Dummy": {
        "estimator": DummyClassifier(strategy="prior"),
        "scale": False
    },
    "LogisticRegression": {
        "estimator": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_STATE
        ),
        "scale": True
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "scale": False
    }
}

all_predictions = []

print("=" * 75)
print("OUT-OF-FOLD PREDICTION GENERATION")
print("=" * 75)

print(f"Samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")
print(f"Folds: {N_SPLITS}")

for target in TARGETS:
    y = y_train[target].astype(int)

    print("\n" + "=" * 75)
    print(f"TARGET: {target}")
    print("=" * 75)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for model_name, config in models.items():
        oof_probabilities = np.full(len(X_train), np.nan)
        fold_assignments = np.full(len(X_train), -1)

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X_train, y),
            start=1
        ):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_train = y.iloc[train_idx]

            if model_name == "Dummy":
                estimator = clone(config["estimator"])
            else:
                estimator = Pipeline([
                    (
                        "preprocessing",
                        build_preprocessor(
                            scale_continuous=config["scale"]
                        )
                    ),
                    ("model", clone(config["estimator"]))
                ])

            estimator.fit(X_fold_train, y_fold_train)

            probabilities = estimator.predict_proba(X_fold_valid)

            positive_class_index = list(estimator.classes_).index(1)

            oof_probabilities[valid_idx] = (
                probabilities[:, positive_class_index]
            )

            fold_assignments[valid_idx] = fold

        if np.isnan(oof_probabilities).any():
            raise ValueError(
                f"Missing OOF predictions: {target}, {model_name}"
            )

        if (fold_assignments == -1).any():
            raise ValueError(
                f"Missing fold assignments: {target}, {model_name}"
            )

        fold_results = pd.DataFrame({
            "sample_index": X_train.index,
            "target": target,
            "model": model_name,
            "fold": fold_assignments,
            "y_true": y.values,
            "y_prob": oof_probabilities
        })

        all_predictions.append(fold_results)

        print(
            f"{model_name:20s} | "
            f"Predictions: {len(oof_probabilities)} | "
            f"Mean probability: {oof_probabilities.mean():.4f}"
        )

oof_df = pd.concat(all_predictions, ignore_index=True)

expected_rows = len(X_train) * len(TARGETS) * len(models)

if len(oof_df) != expected_rows:
    raise ValueError("Unexpected number of OOF predictions.")

if oof_df["y_prob"].isna().any():
    raise ValueError("NaN probabilities found.")

if not oof_df["y_prob"].between(0, 1).all():
    raise ValueError("Probabilities outside [0, 1].")

oof_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 75)
print("OOF GENERATION COMPLETED")
print("=" * 75)

print(f"Total predictions: {len(oof_df)}")
print(f"Expected predictions: {expected_rows}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nPrediction summary:")

print(
    oof_df.groupby(["target", "model"])["y_prob"]
    .agg(["mean", "std", "min", "max"])
    .round(4)
    .to_string()
)
