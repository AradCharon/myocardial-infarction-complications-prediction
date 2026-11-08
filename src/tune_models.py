from pathlib import Path

import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

X_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
Y_PATH = ROOT / "data" / "processed" / "y_train.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics" / "hyperparameter_tuning"

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 20
SCORING = {
    "auprc": "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier": "neg_brier_score",
}


def load_data():
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)

    if len(X) != len(y):
        raise ValueError("X and y have different numbers of rows.")

    y["LET_IS"] = (y["LET_IS"] > 0).astype(int)

    for target in TARGETS:
        if target not in y.columns:
            raise ValueError(f"Target not found: {target}")

        if y[target].isna().any():
            raise ValueError(f"Missing values found in target: {target}")

        if set(y[target].unique()) != {0, 1}:
            raise ValueError(f"Target {target} is not binary.")

        if y[target].value_counts().min() < N_SPLITS:
            raise ValueError(f"Insufficient samples for target: {target}")

    return X, y


def get_model_configurations():
    logistic_regression = {
        "name": "LogisticRegression",
        "model": LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "scale_continuous": True,
        "params": {
            "model__C": loguniform(1e-3, 1e3),
            "model__class_weight": [None, "balanced"],
        },
    }

    random_forest = {
        "name": "RandomForest",
        "model": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "scale_continuous": False,
        "params": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", 0.5, None],
            "model__class_weight": [
                None,
                "balanced",
                "balanced_subsample",
            ],
        },
    }

    return [logistic_regression, random_forest]


def tune_model(X, y, target, configuration):
    preprocessor = build_preprocessor(
        scale_continuous=configuration["scale_continuous"]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", configuration["model"]),
        ]
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=configuration["params"],
        n_iter=N_ITER,
        scoring=SCORING,
        refit="auprc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)

    results_path = OUTPUT_DIR / (
        f"{target}_{configuration['name']}_cv_results.csv"
    )

    results.to_csv(results_path, index=False)

    best_index = search.best_index_

    summary = {
        "target": target,
        "model": configuration["name"],
        "best_cv_auprc": search.cv_results_["mean_test_auprc"][best_index],
        "std_cv_auprc": search.cv_results_["std_test_auprc"][best_index],
        "best_cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best_index],
        "best_cv_brier": -search.cv_results_["mean_test_neg_brier"][best_index],
        "best_params": str(search.best_params_),
        "results_file": results_path.name,
    }

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()

    configurations = get_model_configurations()
    summaries = []

    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    print(f"Training samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Targets: {TARGETS}")
    print(f"CV folds: {N_SPLITS}")
    print(f"Iterations per model: {N_ITER}")

    for target in TARGETS:
        print("\n" + "=" * 70)
        print(f"TARGET: {target}")
        print("=" * 70)

        for configuration in configurations:
            print(f"\nTuning {configuration['name']} for {target}")

            summary = tune_model(
                X=X,
                y=y[target],
                target=target,
                configuration=configuration,
            )

            summaries.append(summary)

            print(f"Best CV AUPRC: {summary['best_cv_auprc']:.4f}")
            print(f"Best CV ROC-AUC: {summary['best_cv_roc_auc']:.4f}")
            print(f"Best CV Brier: {summary['best_cv_brier']:.4f}")
            print(f"Best parameters: {summary['best_params']}")

    summary_df = pd.DataFrame(summaries)

    summary_path = OUTPUT_DIR / "tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 70)

    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()