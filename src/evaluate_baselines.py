
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

X_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
Y_PATH = ROOT / "data" / "processed" / "y_train.csv"
OUTPUT_DIR = ROOT / "outputs" / "metrics"

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

scoring = {
    "auprc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "neg_brier": "neg_brier_score"
}

models = {
    "Dummy": {
        "estimator": DummyClassifier(strategy="prior"),
        "scale": False
    },

    "LogisticRegression": {
        "estimator": LogisticRegression(
            C=1.0,
            penalty="l2",
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

results = []

print("=" * 75)
print("BASELINE MODEL EVALUATION")
print("=" * 75)

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Cross-validation folds: {N_SPLITS}")

for target in TARGETS:

    y = y_train[target].astype(int)

    prevalence = y.mean()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 75)
    print(f"TARGET: {target}")
    print(f"Positive samples: {y.sum()}")
    print(f"Negative samples: {(y == 0).sum()}")
    print(f"Prevalence: {prevalence:.4f}")

    if y.value_counts().min() < N_SPLITS:
        raise ValueError(f"Insufficient samples for CV: {target}")

    for model_name, config in models.items():

        if model_name == "Dummy":
            pipeline = config["estimator"]

        else:
            pipeline = Pipeline([
                (
                    "preprocessing",
                    build_preprocessor(
                        scale_continuous=config["scale"]
                    )
                ),
                ("model", config["estimator"])
            ])

        scores = cross_validate(
            pipeline,
            X_train,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise"
        )

        row = {
            "target": target,
            "model": model_name,
            "prevalence": prevalence
        }

        for metric in scoring:

            values = scores[f"test_{metric}"]

            if metric == "neg_brier":
                values = -values
                metric_name = "brier_score"
            else:
                metric_name = metric

            row[f"{metric_name}_mean"] = np.mean(values)
            row[f"{metric_name}_std"] = np.std(values)

        results.append(row)

        print(
            f"{model_name:20s} | "
            f"AUPRC: {row['auprc_mean']:.4f} | "
            f"ROC-AUC: {row['roc_auc_mean']:.4f} | "
            f"Recall: {row['recall_mean']:.4f} | "
            f"Brier: {row['brier_score_mean']:.4f}"
        )

results_df = pd.DataFrame(results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "baseline_cv_results.csv"

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 75)
print("FINAL RESULTS")
print("=" * 75)

print(
    results_df[
        [
            "target",
            "model",
            "auprc_mean",
            "roc_auc_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "brier_score_mean"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:", output_path)
