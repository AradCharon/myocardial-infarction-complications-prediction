
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

OOF_PATH = ROOT / "outputs" / "predictions" / "oof" / "oof_predictions.csv"
TEST_PATH = ROOT / "outputs" / "predictions" / "final_evaluation" / "test_predictions.csv"

METRICS_DIR = ROOT / "outputs" / "metrics" / "calibration"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions" / "calibration"
PLOTS_DIR = ROOT / "outputs" / "plots" / "calibration"

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]
MODEL_NAME = "RandomForest"

EXPECTED_OOF_SAMPLES = 1360
EXPECTED_TEST_SAMPLES = 340

N_BINS = 10
EPSILON = 1e-6
RANDOM_STATE = 42


def validate_prediction_frame(frame, frame_name):
    required_columns = {
        "target",
        "model",
        "y_true",
        "y_probability",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {frame_name}: {sorted(missing_columns)}"
        )

    frame = frame[
        (frame["model"] == MODEL_NAME)
        & (frame["target"].isin(TARGETS))
    ].copy()

    if frame.empty:
        raise ValueError(f"No valid predictions found in {frame_name}.")

    frame["y_true"] = pd.to_numeric(
        frame["y_true"],
        errors="coerce",
    )

    frame["y_probability"] = pd.to_numeric(
        frame["y_probability"],
        errors="coerce",
    )

    if frame[["y_true", "y_probability"]].isna().any().any():
        raise ValueError(f"Missing or non-numeric values in {frame_name}.")

    if not frame["y_true"].isin([0, 1]).all():
        raise ValueError(f"Invalid target labels in {frame_name}.")

    if not frame["y_probability"].between(0, 1).all():
        raise ValueError(f"Invalid probabilities in {frame_name}.")

    frame["y_true"] = frame["y_true"].astype(int)

    for target in TARGETS:
        target_frame = frame[frame["target"] == target]

        if target_frame.empty:
            raise ValueError(
                f"Missing target {target} in {frame_name}."
            )

        if target_frame["y_true"].nunique() != 2:
            raise ValueError(
                f"Target {target} does not contain both classes in {frame_name}."
            )

        if target_frame["y_probability"].nunique() < 2:
            raise ValueError(
                f"Target {target} has constant probabilities in {frame_name}."
            )

    return frame


def load_data():
    if not OOF_PATH.exists():
        raise FileNotFoundError(OOF_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    oof = pd.read_csv(OOF_PATH)
    test = pd.read_csv(TEST_PATH)

    oof = validate_prediction_frame(oof, "OOF")
    test = validate_prediction_frame(test, "Test")

    for target in TARGETS:
        oof_count = (oof["target"] == target).sum()
        test_count = (test["target"] == target).sum()

        if oof_count != EXPECTED_OOF_SAMPLES:
            raise ValueError(
                f"Unexpected OOF count for {target}: {oof_count}"
            )

        if test_count != EXPECTED_TEST_SAMPLES:
            raise ValueError(
                f"Unexpected Test count for {target}: {test_count}"
            )

    if "sample_index" in oof.columns:
        for target in TARGETS:
            indices = oof.loc[
                oof["target"] == target,
                "sample_index",
            ]

            if indices.duplicated().any():
                raise ValueError(
                    f"Duplicate OOF sample indices for {target}."
                )

    return oof, test


def clip_probabilities(probabilities):
    return np.clip(
        np.asarray(probabilities, dtype=float),
        EPSILON,
        1 - EPSILON,
    )


def probability_to_logit(probabilities):
    probabilities = clip_probabilities(probabilities)

    return np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)


def fit_platt_scaling(y_true, probabilities):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(probabilities, y_true)

    return calibrator


def fit_logit_recalibration(y_true, probabilities):
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        probability_to_logit(probabilities),
        y_true,
    )

    return calibrator


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=N_BINS,
):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = clip_probabilities(probabilities)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities <= bin_edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities < bin_edges[i + 1])
            )

        if not np.any(mask):
            continue

        average_prediction = probabilities[mask].mean()
        average_observed = y_true[mask].mean()

        ece += (
            np.mean(mask)
            * abs(average_prediction - average_observed)
        )

    return float(ece)


def calibration_diagnostics(y_true, probabilities):
    probabilities = clip_probabilities(probabilities)

    diagnostic_model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )

    diagnostic_model.fit(
        probability_to_logit(probabilities),
        y_true,
    )

    return {
        "calibration_intercept": float(
            diagnostic_model.intercept_[0]
        ),
        "calibration_slope": float(
            diagnostic_model.coef_[0][0]
        ),
    }


def calculate_metrics(y_true, probabilities):
    probabilities = clip_probabilities(probabilities)

    prevalence = np.mean(y_true)
    reference_brier = prevalence * (1 - prevalence)

    metrics = {
        "prevalence": float(prevalence),
        "auprc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
        ),
        "brier_skill_score": (
            1 - brier_score_loss(y_true, probabilities) / reference_brier
            if reference_brier > 0
            else np.nan
        ),
    }

    metrics.update(
        calibration_diagnostics(
            y_true,
            probabilities,
        )
    )

    return metrics


def save_reliability_plot(
    target,
    y_true,
    probability_sets,
):
    plt.figure(figsize=(8, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for method_name, probabilities in probability_sets.items():
        observed, predicted = calibration_curve(
            y_true,
            clip_probabilities(probabilities),
            n_bins=N_BINS,
            strategy="uniform",
        )

        plt.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2,
            label=method_name,
        )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.title(f"Reliability Diagram - {target}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / f"{target}_reliability.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_probability_distribution(
    target,
    probability_sets,
):
    plt.figure(figsize=(9, 6))

    for method_name, probabilities in probability_sets.items():
        plt.hist(
            clip_probabilities(probabilities),
            bins=20,
            alpha=0.45,
            density=True,
            label=method_name,
        )

    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title(f"Probability Distribution - {target}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / f"{target}_probability_distribution.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def evaluate_target(target, oof, test):
    oof_target = oof[oof["target"] == target].copy()
    test_target = test[test["target"] == target].copy()

    y_oof = oof_target["y_true"].to_numpy(dtype=int)
    p_oof = oof_target["y_probability"].to_numpy(dtype=float)

    y_test = test_target["y_true"].to_numpy(dtype=int)
    p_test = test_target["y_probability"].to_numpy(dtype=float)

    platt_model = fit_platt_scaling(
        y_oof,
        p_oof,
    )

    logit_model = fit_logit_recalibration(
        y_oof,
        p_oof,
    )

    platt_probabilities = platt_model.predict_proba(
        p_test.reshape(-1, 1)
    )[:, 1]

    logit_probabilities = logit_model.predict_proba(
        probability_to_logit(p_test)
    )[:, 1]

    probability_sets = {
        "Before calibration": clip_probabilities(p_test),
        "Platt Scaling": clip_probabilities(platt_probabilities),
        "Logit Recalibration": clip_probabilities(logit_probabilities),
    }

    metric_rows = []

    for method_name, probabilities in probability_sets.items():
        metrics = calculate_metrics(
            y_test,
            probabilities,
        )

        row = {
            "target": target,
            "model": MODEL_NAME,
            "calibration_method": method_name,
        }

        row.update(metrics)
        metric_rows.append(row)

    parameter_rows = [
        {
            "target": target,
            "method": "Platt Scaling",
            "coefficient": float(platt_model.coef_[0][0]),
            "intercept": float(platt_model.intercept_[0]),
            "input": "raw_probability",
            "training_samples": len(y_oof),
        },
        {
            "target": target,
            "method": "Logit Recalibration",
            "coefficient": float(logit_model.coef_[0][0]),
            "intercept": float(logit_model.intercept_[0]),
            "input": "logit_probability",
            "training_samples": len(y_oof),
        },
    ]

    prediction_frame = pd.DataFrame(
        {
            "target": target,
            "model": MODEL_NAME,
            "y_true": y_test,
            "raw_probability": probability_sets["Before calibration"],
            "platt_probability": probability_sets["Platt Scaling"],
            "logit_recalibrated_probability": probability_sets[
                "Logit Recalibration"
            ],
        }
    )

    save_reliability_plot(
        target,
        y_test,
        probability_sets,
    )

    save_probability_distribution(
        target,
        probability_sets,
    )

    return metric_rows, parameter_rows, prediction_frame


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    oof, test = load_data()

    metric_rows = []
    parameter_rows = []
    prediction_frames = []

    print("=" * 75)
    print("PROBABILITY CALIBRATION ANALYSIS")
    print("=" * 75)

    print(f"Model: {MODEL_NAME}")
    print(f"OOF samples per target: {EXPECTED_OOF_SAMPLES}")
    print(f"Test samples per target: {EXPECTED_TEST_SAMPLES}")
    print(f"Calibration bins: {N_BINS}")

    for target in TARGETS:
        print("\n" + "=" * 75)
        print(f"TARGET: {target}")
        print("=" * 75)

        target_metrics, target_parameters, target_predictions = (
            evaluate_target(
                target,
                oof,
                test,
            )
        )

        metric_rows.extend(target_metrics)
        parameter_rows.extend(target_parameters)
        prediction_frames.append(target_predictions)

        target_metrics_df = pd.DataFrame(target_metrics)

        display_columns = [
            "calibration_method",
            "auprc",
            "roc_auc",
            "brier_score",
            "log_loss",
            "ece",
            "brier_skill_score",
            "calibration_intercept",
            "calibration_slope",
        ]

        print(
            target_metrics_df[display_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        print("\nCALIBRATION PARAMETERS")

        for parameter in target_parameters:
            print(
                f"{parameter['method']}: "
                f"A={parameter['coefficient']:.6f}, "
                f"B={parameter['intercept']:.6f}"
            )

    metrics_df = pd.DataFrame(metric_rows)
    parameters_df = pd.DataFrame(parameter_rows)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    metrics_path = METRICS_DIR / "calibration_comparison.csv"
    parameters_path = METRICS_DIR / "calibration_parameters.csv"
    predictions_path = (
        PREDICTIONS_DIR / "calibrated_test_predictions.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    parameters_df.to_csv(
        parameters_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print("CALIBRATION ANALYSIS COMPLETED")
    print("=" * 75)

    print(f"Metrics: {metrics_path}")
    print(f"Parameters: {parameters_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
