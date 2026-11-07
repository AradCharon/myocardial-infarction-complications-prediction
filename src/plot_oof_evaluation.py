
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "outputs" / "predictions" / "oof_predictions.csv"

FIGURE_DIR = ROOT / "outputs" / "figures" / "evaluation"
METRIC_DIR = ROOT / "outputs" / "metrics"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]

MODELS = [
    "Dummy",
    "LogisticRegression",
    "RandomForest"
]

N_BINS = 10
THRESHOLD = 0.5

oof_df = pd.read_csv(INPUT_PATH)

required_columns = {
    "sample_index",
    "target",
    "model",
    "fold",
    "y_true",
    "y_prob"
}

if not required_columns.issubset(oof_df.columns):
    raise ValueError("Required columns are missing.")

if oof_df["y_prob"].isna().any():
    raise ValueError("Missing probabilities found.")

if not oof_df["y_prob"].between(0, 1).all():
    raise ValueError("Invalid probability values.")

metrics_results = []

for target in TARGETS:

    target_df = oof_df[oof_df["target"] == target]

    if target_df.empty:
        raise ValueError(f"No predictions found for {target}.")

    expected_samples = target_df["sample_index"].nunique()

    if expected_samples == 0:
        raise ValueError(f"No samples found for {target}.")

    model_data = {}

    for model in MODELS:

        model_df = target_df[
            target_df["model"] == model
        ].sort_values("sample_index").reset_index(drop=True)

        if len(model_df) != expected_samples:
            raise ValueError(
                f"Unexpected prediction count: {target}, {model}"
            )

        if model_df["sample_index"].duplicated().any():
            raise ValueError(
                f"Duplicate sample indices: {target}, {model}"
            )

        model_data[model] = model_df

    reference_indices = model_data[MODELS[0]]["sample_index"].to_numpy()
    y_true = model_data[MODELS[0]]["y_true"].to_numpy()

    if len(np.unique(y_true)) != 2:
        raise ValueError(f"Target must contain both classes: {target}")

    for model in MODELS:

        model_df = model_data[model]

        if not np.array_equal(
            reference_indices,
            model_df["sample_index"].to_numpy()
        ):
            raise ValueError(
                f"Sample indices do not match: {target}, {model}"
            )

        if not np.array_equal(
            y_true,
            model_df["y_true"].to_numpy()
        ):
            raise ValueError(
                f"Target values do not match: {target}, {model}"
            )

        y_prob = model_df["y_prob"].to_numpy()

        metrics_results.append({
            "target": target,
            "model": model,
            "prevalence": y_true.mean(),
            "auprc": average_precision_score(y_true, y_prob),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob),
            "precision_at_0.5": precision_score(
                y_true,
                y_prob >= THRESHOLD,
                zero_division=0
            ),
            "recall_at_0.5": recall_score(
                y_true,
                y_prob >= THRESHOLD,
                zero_division=0
            ),
            "f1_at_0.5": f1_score(
                y_true,
                y_prob >= THRESHOLD,
                zero_division=0
            )
        })

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in MODELS:

        y_prob = model_data[model]["y_prob"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model} (AUC = {roc_auc:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title(f"ROC Curve - {target}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{target}_roc_curve.png",
        dpi=300
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in MODELS:

        y_prob = model_data[model]["y_prob"].to_numpy()

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob
        )

        auprc = average_precision_score(y_true, y_prob)

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model} (AP = {auprc:.3f})"
        )

    ax.axhline(
        y=y_true.mean(),
        linestyle="--",
        label=f"Prevalence = {y_true.mean():.3f}"
    )

    ax.set_title(f"Precision-Recall Curve - {target}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{target}_precision_recall_curve.png",
        dpi=300
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in MODELS:

        y_prob = model_data[model]["y_prob"].to_numpy()

        fraction_positive, mean_predicted = calibration_curve(
            y_true,
            y_prob,
            n_bins=N_BINS,
            strategy="quantile"
        )

        ax.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            linewidth=2,
            label=model
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    ax.set_title(f"Calibration Curve - {target}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{target}_calibration_curve.png",
        dpi=300
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in MODELS:

        y_prob = model_data[model]["y_prob"].to_numpy()

        ax.hist(
            y_prob,
            bins=30,
            alpha=0.4,
            density=True,
            label=model
        )

    ax.set_title(f"Predicted Probability Distribution - {target}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{target}_probability_distribution.png",
        dpi=300
    )
    plt.close(fig)


metrics_df = pd.DataFrame(metrics_results)

metrics_path = METRIC_DIR / "oof_evaluation_metrics.csv"

metrics_df.to_csv(
    metrics_path,
    index=False
)

print("=" * 75)
print("OOF EVALUATION COMPLETED")
print("=" * 75)

print("\nMETRICS")

print(
    metrics_df.round(4).to_string(index=False)
)

print("\nFigures saved to:")
print(FIGURE_DIR)

print("\nMetrics saved to:")
print(metrics_path)
