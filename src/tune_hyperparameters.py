
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

X_PATH = ROOT / "data" / "processed" / "selected" / "X_train.csv"
Y_PATH = ROOT / "data" / "processed" / "y_train.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["LET_IS", "ZSN", "FIBR_PREDS"]

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 25

X_train = pd.read_csv(X_PATH)
y_train = pd.read_csv(Y_PATH)

if len(X_train) != len(y_train):
    raise ValueError("Train datasets have different lengths.")

y_train["LET_IS"] = (y_train["LET_IS"] > 0).astype(int)

models = {
    "LogisticRegression": {
        "pipeline": Pipeline([
            ("preprocessing", build_preprocessor(scale_continuous=True)),
            ("model", LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            ))
        ]),
        "params": {
            "model__C": np.logspace(-3, 2, 30),
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"],
            "model__penalty": ["l2"]
        }
    },
    "RandomForest": {
        "pipeline": Pipeline([
            ("preprocessing", build_preprocessor(scale_continuous=False)),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "params": {
            "model__n_estimators": [300, 500, 700, 1000],
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__class_weight": [None, "balanced", "balanced_subsample"]
        }
    }
}

all_results = []
best_params = {}

for target in TARGETS:
    y = y_train[target].astype(int)

    if y.value_counts().min() < N_SPLITS:
        raise ValueError(f"Insufficient samples for {target}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 75)
    print(f"TARGET: {target}")
    print("=" * 75)

    for model_name, config in models.items():
        search = RandomizedSearchCV(
            estimator=clone(config["pipeline"]),
            param_distributions=config["params"],
            n_iter=N_ITER,
            scoring="average_precision",
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score="raise"
        )

        search.fit(X_train, y)

        best_params[f"{target}_{model_name}"] = search.best_params_

        all_results.append({
            "target": target,
            "model": model_name,
            "best_auprc": search.best_score_,
            "train_auprc": search.cv_results_["mean_train_score"][search.best_index_],
            "best_params": json.dumps(search.best_params_)
        })

        print(f"\nModel: {model_name}")
        print(f"Best AUPRC: {search.best_score_:.4f}")
        print(f"Train AUPRC: {search.cv_results_['mean_train_score'][search.best_index_]:.4f}")
        print(f"Best parameters: {search.best_params_}")

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_DIR / "hyperparameter_tuning_results.csv",
    index=False
)

with open(
    OUTPUT_DIR / "best_hyperparameters.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(best_params, file, indent=4)

print("\n" + "=" * 75)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 75)

print(results_df.to_string(index=False))
