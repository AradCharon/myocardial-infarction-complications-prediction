
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "outputs" / "predictions" / "oof_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]

MODELS = [
    "Dummy",
    "LogisticRegression",
    "RandomForest"
]

THRESHOLDS = np.arange(0.01, 1.00, 0.01)

oof_df = pd.read_csv(INPUT_PATH)

threshold_results = []
best_results = []

for target in TARGETS:

    for model in MODELS:

        subset = oof_df[
            (oof_df["target"] == target) &
            (oof_df["model"] == model)
        ].sort_values("sample_index")

        y_true = subset["y_true"].to_numpy()
        y_prob = subset["y_prob"].to_numpy()

        for threshold in THRESHOLDS:

            y_pred = (y_prob >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            ).ravel()

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f2 = fbeta_score(
                y_true,
                y_pred,
                beta=2,
                zero_division=0
            )

            threshold_results.append({
                "target": target,
                "model": model,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "f2": f2,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn
            })

threshold_df = pd.DataFrame(threshold_results)

for target in TARGETS:

    for model in MODELS:

        subset = threshold_df[
            (threshold_df["target"] == target) &
            (threshold_df["model"] == model)
        ]

        best_f1 = subset.loc[subset["f1"].idxmax()]
        best_f2 = subset.loc[subset["f2"].idxmax()]

        best_results.append({
            "target": target,
            "model": model,
            "criterion": "F1",
            "threshold": best_f1["threshold"],
            "precision": best_f1["precision"],
            "recall": best_f1["recall"],
            "specificity": best_f1["specificity"],
            "f1": best_f1["f1"],
            "f2": best_f1["f2"]
        })

        best_results.append({
            "target": target,
            "model": model,
            "criterion": "F2",
            "threshold": best_f2["threshold"],
            "precision": best_f2["precision"],
            "recall": best_f2["recall"],
            "specificity": best_f2["specificity"],
            "f1": best_f2["f1"],
            "f2": best_f2["f2"]
        })

threshold_df.to_csv(
    OUTPUT_DIR / "threshold_analysis.csv",
    index=False
)

best_df = pd.DataFrame(best_results)

best_df.to_csv(
    OUTPUT_DIR / "best_thresholds.csv",
    index=False
)

print("=" * 75)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 75)

print("\nBEST THRESHOLDS")

print(
    best_df.round(4).to_string(index=False)
)

print("\nFull threshold results saved to:")
print(OUTPUT_DIR / "threshold_analysis.csv")

print("\nBest thresholds saved to:")
print(OUTPUT_DIR / "best_thresholds.csv")
