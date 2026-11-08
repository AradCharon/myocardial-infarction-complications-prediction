
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

X_TRAIN_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
X_TEST_PATH = ROOT / "data" / "processed" / "selected" / "X_test.csv"

Y_TRAIN_PATH = ROOT / "data" / "processed" / "y_train.csv"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics" / "final_evaluation"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions" / "final_evaluation"

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]

RANDOM_STATE = 42
THRESHOLD = 0.5


MODEL_CONFIGS = {
    "LET_IS": {
        "LogisticRegression": {
            "type": "logistic",
            "C": 0.6317967091932509,
            "class_weight": None,
        },
        "RandomForest": {
            "type": "forest",
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    },
    "ZSN": {
        "LogisticRegression": {
            "type": "logistic",
            "C": 0.10051981180656774,
            "class_weight": "balanced",
        },
        "RandomForest": {
            "type": "forest",
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 2,
            "max_features": None,
            "class_weight": "balanced",
        },
    },
    "FIBR_PREDS": {
        "LogisticRegression": {
            "type": "logistic",
            "C": 17.71884735480682,
            "class_weight": "balanced",
        },
        "RandomForest": {
            "type": "forest",
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    },
}


def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test features do not match.")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")

    y_train["LET_IS"] = (y_train["LET_IS"] > 0).astype(int)
    y_test["LET_IS"] = (y_test["LET_IS"] > 0).astype(int)

    for target in TARGETS:
        if target not in y_train.columns or target not in y_test.columns:
            raise ValueError(f"Missing target: {target}")

        if y_train[target].isna().any() or y_test[target].isna().any():
            raise ValueError(f"Missing labels in target: {target}")

        if set(y_train[target].unique()) != {0, 1}:
            raise ValueError(f"Training target {target} is not binary.")

        if set(y_test[target].unique()) != {0, 1}:
            raise ValueError(f"Test target {target} is not binary.")

    return X_train, X_test, y_train, y_test


def build_model(model_name, target):
    if model_name == "Dummy":
        return DummyClassifier(strategy="prior")

    config = MODEL_CONFIGS[target][model_name]

    if config["type"] == "logistic":
        estimator = LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        scale_continuous = True

    elif config["type"] == "forest":
        estimator = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            class_weight=config["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scale_continuous = False

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(scale_continuous=scale_continuous),
            ),
            ("model", estimator),
        ]
    )


def evaluate_predictions(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": THRESHOLD,
        "prevalence": float(np.mean(y_true)),
        "auprc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
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
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    results = []
    prediction_frames = []

    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    for target in TARGETS:
        print("\n" + "=" * 70)
        print(f"TARGET: {target}")
        print("=" * 70)

        y_train_target = y_train[target]
        y_test_target = y_test[target]

        model_names = ["Dummy", "LogisticRegression", "RandomForest"]

        for model_name in model_names:
            print(f"\nEvaluating {model_name}...")

            model = build_model(model_name, target)

            model.fit(X_train, y_train_target)

            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_predictions(
                y_test_target,
                probabilities,
            )

            metrics["target"] = target
            metrics["model"] = model_name

            results.append(metrics)

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "model": model_name,
                        "y_true": y_test_target.to_numpy(),
                        "y_probability": probabilities,
                        "y_pred": (
                            probabilities >= THRESHOLD
                        ).astype(int),
                    }
                )
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

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "target",
            "model",
            "prevalence",
            "threshold",
            "auprc",
            "roc_auc",
            "brier_score",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]

    results_path = OUTPUT_DIR / "test_metrics.csv"
    predictions_path = PREDICTIONS_DIR / "test_predictions.csv"

    results_df.to_csv(results_path, index=False)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETED")
    print("=" * 70)

    print(results_df.to_string(index=False))

    print(f"\nMetrics saved to: {results_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
