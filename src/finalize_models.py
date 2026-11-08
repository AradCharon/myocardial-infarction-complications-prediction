
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

X_TRAIN_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
X_TEST_PATH = ROOT / "data" / "processed" / "selected" / "X_test.csv"
Y_TRAIN_PATH = ROOT / "data" / "processed" / "y_train.csv"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.csv"

OOF_CANDIDATES = [
    ROOT / "outputs" / "predictions" / "oof" / "oof_predictions.csv",
    ROOT / "outputs" / "predictions" / "oof_predictions.csv",
]

OUTPUT_METRICS = ROOT / "outputs" / "metrics" / "finalization"
OUTPUT_PREDICTIONS = ROOT / "outputs" / "predictions" / "finalization"
OUTPUT_PLOTS = ROOT / "outputs" / "plots" / "finalization"

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]

MODEL_NAME = "RandomForest"
RANDOM_STATE = 42
EPSILON = 1e-6
N_BINS = 10

RF_PARAMS = {
    "LET_IS": {
        "n_estimators": 300,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_depth": None,
        "class_weight": "balanced",
    },
    "ZSN": {
        "n_estimators": 500,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": None,
        "max_depth": 10,
        "class_weight": "balanced",
    },
    "FIBR_PREDS": {
        "n_estimators": 300,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_depth": None,
        "class_weight": "balanced",
    },
}


def load_data():
    required_paths = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns do not match.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths do not match.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test lengths do not match.")

    if "LET_IS" in y_train.columns:
        y_train["LET_IS"] = (y_train["LET_IS"] > 0).astype(int)
        y_test["LET_IS"] = (y_test["LET_IS"] > 0).astype(int)

    for target in TARGETS:
        if target not in y_train.columns or target not in y_test.columns:
            raise ValueError(f"Missing target: {target}")

        if y_train[target].isna().any() or y_test[target].isna().any():
            raise ValueError(f"Missing target values: {target}")

        if not set(y_train[target].unique()).issubset({0, 1}):
            raise ValueError(f"Training target is not binary: {target}")

        if not set(y_test[target].unique()).issubset({0, 1}):
            raise ValueError(f"Test target is not binary: {target}")

    return X_train, X_test, y_train, y_test


def load_oof_predictions():
    oof_path = next(
        (path for path in OOF_CANDIDATES if path.exists()),
        None,
    )

    if oof_path is None:
        raise FileNotFoundError(
            "OOF predictions file was not found."
        )

    oof = pd.read_csv(oof_path)

    if "y_probability" in oof.columns:
        oof = oof.rename(
            columns={"y_probability": "y_prob"}
        )

    required = {
        "target",
        "model",
        "y_true",
        "y_prob",
        "fold",
    }

    missing = required - set(oof.columns)

    if missing:
        raise ValueError(
            f"Missing OOF columns: {sorted(missing)}"
        )

    oof = oof[
        (oof["model"] == MODEL_NAME)
        & (oof["target"].isin(TARGETS))
    ].copy()

    if oof.empty:
        raise ValueError("No RandomForest OOF predictions found.")

    if oof["y_prob"].isna().any():
        raise ValueError("OOF predictions contain missing probabilities.")

    if not oof["y_prob"].between(0, 1).all():
        raise ValueError("OOF probabilities outside [0, 1].")

    return oof


def build_model(target):
    estimator = RandomForestClassifier(
        **RF_PARAMS[target],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(scale_continuous=False),
            ),
            ("model", estimator),
        ]
    )


def clipped_probabilities(probabilities):
    return np.clip(
        np.asarray(probabilities, dtype=float),
        EPSILON,
        1 - EPSILON,
    )


def probability_logit(probabilities):
    p = clipped_probabilities(probabilities)
    return np.log(p / (1 - p))


def fit_calibrator(y_true, probabilities, method):
    probabilities = clipped_probabilities(probabilities)

    if method == "Platt":
        calibration_input = probabilities.reshape(-1, 1)
    elif method == "Logit":
        calibration_input = probability_logit(
            probabilities
        ).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(calibration_input, y_true)

    return calibrator


def apply_calibrator(calibrator, probabilities, method):
    probabilities = clipped_probabilities(probabilities)

    if method == "Platt":
        calibration_input = probabilities.reshape(-1, 1)
    elif method == "Logit":
        calibration_input = probability_logit(
            probabilities
        ).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return calibrator.predict_proba(calibration_input)[:, 1]


def expected_calibration_error(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0, 1, N_BINS + 1)
    ece = 0.0

    for index in range(N_BINS):
        if index == N_BINS - 1:
            mask = (
                (probabilities >= bin_edges[index])
                & (probabilities <= bin_edges[index + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[index])
                & (probabilities < bin_edges[index + 1])
            )

        if mask.sum() == 0:
            continue

        observed = y_true[mask].mean()
        predicted = probabilities[mask].mean()

        ece += (
            mask.sum()
            / len(y_true)
            * abs(observed - predicted)
        )

    return float(ece)


def calibration_metrics(y_true, probabilities):
    probabilities = clipped_probabilities(probabilities)

    prevalence = np.mean(y_true)
    reference_brier = brier_score_loss(
        y_true,
        np.full(len(y_true), prevalence),
    )

    brier = brier_score_loss(y_true, probabilities)

    bss = (
        1 - brier / reference_brier
        if reference_brier > 0
        else np.nan
    )

    return {
        "auprc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier,
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
        ),
        "brier_skill_score": bss,
    }


def crossfit_calibration(oof_target):
    y_true = oof_target["y_true"].to_numpy(dtype=int)
    probabilities = oof_target["y_prob"].to_numpy(dtype=float)
    folds = oof_target["fold"].to_numpy()

    methods = ["Platt", "Logit"]
    results = {}

    for method in methods:
        calibrated = np.full(len(oof_target), np.nan)

        for fold in np.unique(folds):
            train_mask = folds != fold
            valid_mask = folds == fold

            if len(np.unique(y_true[train_mask])) < 2:
                raise ValueError(
                    f"Insufficient classes for calibration: {fold}"
                )

            calibrator = fit_calibrator(
                y_true[train_mask],
                probabilities[train_mask],
                method,
            )

            calibrated[valid_mask] = apply_calibrator(
                calibrator,
                probabilities[valid_mask],
                method,
            )

        if np.isnan(calibrated).any():
            raise ValueError(
                f"Missing cross-fitted probabilities: {method}"
            )

        results[method] = calibrated

    return results


def select_calibration_method(y_true, calibrated_oof):
    method_metrics = {}

    for method, probabilities in calibrated_oof.items():
        method_metrics[method] = calibration_metrics(
            y_true,
            probabilities,
        )

    selected_method = min(
        method_metrics,
        key=lambda method: (
            method_metrics[method]["brier_score"],
            method_metrics[method]["log_loss"],
        ),
    )

    return selected_method, method_metrics


def threshold_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        ),
        "youden_j": recall + specificity - 1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_thresholds(y_true, probabilities):
    thresholds = np.unique(
        np.concatenate([
            np.arange(0.01, 1.00, 0.01),
            np.asarray(probabilities),
        ])
    )

    rows = []

    for threshold in thresholds:
        rows.append(
            threshold_metrics(
                y_true,
                probabilities,
                threshold,
            )
        )

    threshold_frame = pd.DataFrame(rows)

    best = {}

    for criterion in ["f1", "f2", "youden_j"]:
        best_row = threshold_frame.loc[
            threshold_frame[criterion].idxmax()
        ]

        best[criterion] = best_row.to_dict()

    return threshold_frame, best


def evaluate_final(y_true, probabilities, threshold):
    result = threshold_metrics(
        y_true,
        probabilities,
        threshold,
    )

    result.update(
        calibration_metrics(
            y_true,
            probabilities,
        )
    )

    return result


def save_reliability_plot(target, y_true, raw, calibrated):
    plt.figure(figsize=(8, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for label, probabilities in [
        ("Before calibration", raw),
        ("After calibration", calibrated),
    ]:
        bin_edges = np.linspace(0, 1, N_BINS + 1)

        mean_predicted = []
        fraction_positive = []

        for index in range(N_BINS):
            if index == N_BINS - 1:
                mask = (
                    (probabilities >= bin_edges[index])
                    & (probabilities <= bin_edges[index + 1])
                )
            else:
                mask = (
                    (probabilities >= bin_edges[index])
                    & (probabilities < bin_edges[index + 1])
                )

            if mask.sum() == 0:
                continue

            mean_predicted.append(
                probabilities[mask].mean()
            )

            fraction_positive.append(
                y_true[mask].mean()
            )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=label,
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(f"Final Reliability Diagram - {target}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOTS / f"{target}_final_reliability.png",
        dpi=200,
    )

    plt.close()


def save_confusion_plot(target, y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
            )

    ax.set_title(f"Confusion Matrix - {target}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOTS / f"{target}_confusion_matrix.png",
        dpi=200,
    )

    plt.close(fig)


def save_error_analysis(target, X_test, y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    errors = X_test.copy()

    errors.insert(
        0,
        "sample_index",
        X_test.index,
    )

    errors["target"] = target
    errors["y_true"] = y_true
    errors["y_probability"] = probabilities
    errors["threshold"] = threshold
    errors["y_pred"] = predictions

    errors["error_type"] = np.select(
        [
            (y_true == 0) & (predictions == 1),
            (y_true == 1) & (predictions == 0),
        ],
        [
            "False Positive",
            "False Negative",
        ],
        default="Correct",
    )

    errors["prediction_confidence"] = np.where(
        predictions == 1,
        probabilities,
        1 - probabilities,
    )

    errors.to_csv(
        OUTPUT_PREDICTIONS / f"{target}_error_analysis.csv",
        index=False,
    )


def save_feature_importance(target, model, X_test, y_test):
    fitted_preprocessor = model.named_steps["preprocessing"]
    fitted_rf = model.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    if len(feature_names) != len(fitted_rf.feature_importances_):
        raise ValueError(
            f"Feature name mismatch for {target}."
        )

    native_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": fitted_rf.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    native_importance.to_csv(
        OUTPUT_METRICS / f"{target}_feature_importance.csv",
        index=False,
    )

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    permutation_frame = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }).sort_values(
        "importance_mean",
        ascending=False,
    )

    permutation_frame.to_csv(
        OUTPUT_METRICS / f"{target}_permutation_importance.csv",
        index=False,
    )

    top_features = permutation_frame.head(20).sort_values(
        "importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
    )

    plt.xlabel("Mean decrease in average precision")
    plt.title(f"Permutation Importance - {target}")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOTS / f"{target}_permutation_importance.png",
        dpi=200,
    )

    plt.close()


def main():
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    oof = load_oof_predictions()

    final_metrics = []
    calibration_results = []
    threshold_results = []
    prediction_frames = []
    parameter_results = []

    print("=" * 75)
    print("FINAL MODEL PIPELINE")
    print("=" * 75)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    for target in TARGETS:
        print("\n" + "=" * 75)
        print(f"TARGET: {target}")
        print("=" * 75)

        oof_target = oof[
            oof["target"] == target
        ].copy()

        oof_target = oof_target.reset_index(drop=True)

        if len(oof_target) != len(X_train):
            raise ValueError(
                f"Unexpected OOF count for {target}: "
                f"{len(oof_target)}"
            )

        y_oof = oof_target["y_true"].to_numpy(dtype=int)
        raw_oof = oof_target["y_prob"].to_numpy(dtype=float)

        if not np.array_equal(
            y_oof,
            y_train[target].to_numpy(dtype=int),
        ):
            raise ValueError(
                f"OOF labels are not aligned with training labels: {target}"
            )

        crossfitted = crossfit_calibration(oof_target)

        selected_method, method_metrics = select_calibration_method(
            y_oof,
            crossfitted,
        )

        selected_oof = crossfitted[selected_method]

        for method, metrics in method_metrics.items():
            calibration_results.append({
                "target": target,
                "method": method,
                **metrics,
            })

        print("\nCALIBRATION COMPARISON")

        for method, metrics in method_metrics.items():
            print(
                f"{method:10s} | "
                f"Brier: {metrics['brier_score']:.4f} | "
                f"Log Loss: {metrics['log_loss']:.4f} | "
                f"ECE: {metrics['ece']:.4f}"
            )

        print(f"Selected calibration: {selected_method}")

        _, best_thresholds = find_best_thresholds(
            y_oof,
            selected_oof,
        )

        for criterion, result in best_thresholds.items():
            threshold_results.append({
                "target": target,
                "criterion": criterion,
                **result,
            })

        selected_threshold = float(
            best_thresholds["f2"]["threshold"]
        )

        print("\nSELECTED THRESHOLDS")

        for criterion, result in best_thresholds.items():
            print(
                f"{criterion:10s} | "
                f"Threshold: {result['threshold']:.4f} | "
                f"Precision: {result['precision']:.4f} | "
                f"Recall: {result['recall']:.4f} | "
                f"Specificity: {result['specificity']:.4f}"
            )

        final_calibrator = fit_calibrator(
            y_oof,
            raw_oof,
            selected_method,
        )

        parameter_results.append({
            "target": target,
            "method": selected_method,
            "A": float(final_calibrator.coef_[0, 0]),
            "B": float(final_calibrator.intercept_[0]),
        })

        model = build_model(target)

        model.fit(
            X_train,
            y_train[target].astype(int),
        )

        raw_test = model.predict_proba(X_test)[:, 1]

        calibrated_test = apply_calibrator(
            final_calibrator,
            raw_test,
            selected_method,
        )

        y_test_target = y_test[target].to_numpy(dtype=int)

        raw_test_metrics = evaluate_final(
            y_test_target,
            raw_test,
            0.5,
        )

        calibrated_test_metrics = evaluate_final(
            y_test_target,
            calibrated_test,
            selected_threshold,
        )

        final_metrics.append({
            "target": target,
            "model": MODEL_NAME,
            "calibration_method": selected_method,
            "threshold": selected_threshold,
            "evaluation": "Calibrated",
            **calibrated_test_metrics,
        })

        final_metrics.append({
            "target": target,
            "model": MODEL_NAME,
            "calibration_method": "None",
            "threshold": 0.5,
            "evaluation": "Raw baseline",
            **raw_test_metrics,
        })

        prediction_frames.append(
            pd.DataFrame({
                "target": target,
                "y_true": y_test_target,
                "raw_probability": raw_test,
                "calibrated_probability": calibrated_test,
                "threshold": selected_threshold,
                "y_pred": (
                    calibrated_test >= selected_threshold
                ).astype(int),
            })
        )

        save_reliability_plot(
            target,
            y_test_target,
            raw_test,
            calibrated_test,
        )

        save_confusion_plot(
            target,
            y_test_target,
            calibrated_test,
            selected_threshold,
        )

        save_error_analysis(
            target,
            X_test,
            y_test_target,
            calibrated_test,
            selected_threshold,
        )

        save_feature_importance(
            target,
            model,
            X_test,
            y_test_target,
        )

        print("\nFINAL TEST RESULTS")

        print(
            f"AUPRC: {calibrated_test_metrics['auprc']:.4f}"
        )

        print(
            f"ROC-AUC: {calibrated_test_metrics['roc_auc']:.4f}"
        )

        print(
            f"Brier: {calibrated_test_metrics['brier_score']:.4f}"
        )

        print(
            f"Precision: {calibrated_test_metrics['precision']:.4f}"
        )

        print(
            f"Recall: {calibrated_test_metrics['recall']:.4f}"
        )

        print(
            f"Specificity: {calibrated_test_metrics['specificity']:.4f}"
        )

        print(
            f"F1: {calibrated_test_metrics['f1']:.4f}"
        )

        print(
            f"F2: {calibrated_test_metrics['f2']:.4f}"
        )

    pd.DataFrame(final_metrics).to_csv(
        OUTPUT_METRICS / "final_test_metrics.csv",
        index=False,
    )

    pd.DataFrame(calibration_results).to_csv(
        OUTPUT_METRICS / "crossfitted_calibration_comparison.csv",
        index=False,
    )

    pd.DataFrame(threshold_results).to_csv(
        OUTPUT_METRICS / "selected_thresholds.csv",
        index=False,
    )

    pd.DataFrame(parameter_results).to_csv(
        OUTPUT_METRICS / "final_calibration_parameters.csv",
        index=False,
    )

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_PREDICTIONS / "final_test_predictions.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("FINALIZATION COMPLETED")
    print("=" * 75)

    print(f"Metrics: {OUTPUT_METRICS}")
    print(f"Predictions: {OUTPUT_PREDICTIONS}")
    print(f"Plots: {OUTPUT_PLOTS}")

    print("\nFINAL TEST METRICS")

    print(
        pd.DataFrame(final_metrics)[
            [
                "target",
                "evaluation",
                "calibration_method",
                "threshold",
                "auprc",
                "roc_auc",
                "brier_score",
                "precision",
                "recall",
                "specificity",
                "f1",
                "f2",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
