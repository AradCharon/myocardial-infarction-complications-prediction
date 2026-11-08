
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evaluate_final_models import (
    build_model,
    TARGETS,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    RANDOM_STATE,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "predictions" / "oof"
METRICS_DIR = ROOT / "outputs" / "metrics" / "oof"

N_SPLITS = 5
THRESHOLD = 0.5

MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
]


def load_training_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    if len(X_train) != len(y_train):
        raise ValueError("Feature and target row counts do not match.")

    if "LET_IS" in y_train.columns:
        y_train["LET_IS"] = (y_train["LET_IS"] > 0).astype(int)

    for target in TARGETS:
        if target not in y_train.columns:
            raise ValueError(f"Missing target: {target}")

        if y_train[target].isna().any():
            raise ValueError(f"Missing labels in target: {target}")

        if set(y_train[target].unique()) != {0, 1}:
            raise ValueError(f"Target {target} is not binary.")

    return X_train, y_train


def evaluate_oof_predictions(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "prevalence": float(np.mean(y_true)),
        "threshold": THRESHOLD,
        "auprc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_training_data()

    prediction_frames = []
    metric_rows = []

    print("=" * 70)
    print("OUT-OF-FOLD PREDICTION GENERATION")
    print("=" * 70)

    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Folds: {N_SPLITS}")

    for target in TARGETS:
        print("\n" + "=" * 70)
        print(f"TARGET: {target}")
        print("=" * 70)

        y_target = y_train[target].reset_index(drop=True)

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        cv_splits = list(cv.split(X_train, y_target))

        fold_ids = np.full(
            len(X_train),
            -1,
            dtype=int,
        )

        for fold_number, (_, validation_indices) in enumerate(cv_splits):
            fold_ids[validation_indices] = fold_number

        if np.any(fold_ids < 0):
            raise ValueError(f"Some samples have no fold assignment: {target}")

        if len(np.unique(fold_ids)) != N_SPLITS:
            raise ValueError(f"Unexpected fold count: {target}")

        print(f"Positive samples: {int(y_target.sum())}")
        print(f"Negative samples: {int((y_target == 0).sum())}")

        for model_name in MODEL_NAMES:
            print(f"\nGenerating OOF predictions: {model_name}")

            model = build_model(model_name, target)

            probabilities = cross_val_predict(
                model,
                X_train,
                y_target,
                cv=cv_splits,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]

            if len(probabilities) != len(X_train):
                raise ValueError(
                    f"Unexpected prediction count: {target}, {model_name}"
                )

            if not np.isfinite(probabilities).all():
                raise ValueError(
                    f"Invalid OOF probabilities: {target}, {model_name}"
                )

            if not np.all(
                (probabilities >= 0) & (probabilities <= 1)
            ):
                raise ValueError(
                    f"OOF probabilities outside [0, 1]: {target}, {model_name}"
                )

            metrics = evaluate_oof_predictions(
                y_target.to_numpy(),
                probabilities,
            )

            metrics["target"] = target
            metrics["model"] = model_name

            metric_rows.append(metrics)

            prediction_frames.append(
                pd.DataFrame({
                    "sample_index": X_train.index,
                    "target": target,
                    "model": model_name,
                    "fold": fold_ids,
                    "y_true": y_target.to_numpy(),
                    "y_probability": probabilities,
                    "y_pred_05": (
                        probabilities >= THRESHOLD
                    ).astype(int),
                })
            )

            print(
                f"AUPRC: {metrics['auprc']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"Brier: {metrics['brier_score']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"Specificity: {metrics['specificity']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(metric_rows)

    predictions_path = OUTPUT_DIR / "oof_predictions.csv"
    metrics_path = METRICS_DIR / "oof_metrics.csv"

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("OOF GENERATION COMPLETED")
    print("=" * 70)

    print(metrics_df.to_string(index=False))

    print("\nFOLD VALIDATION")

    print(
        predictions_df.groupby(
            ["target", "model", "fold"]
        ).size().to_string()
    )

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
