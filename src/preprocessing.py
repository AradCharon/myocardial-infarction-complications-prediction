
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "selected_feature_type_profile.csv"
)


def get_feature_groups():
    """
    Load the validated feature mapping and
    return feature names grouped by type.
    """

    profile = pd.read_csv(PROFILE_PATH)

    binary_features = profile.loc[
        profile["feature_type"] == "binary", "feature"
    ].tolist()

    ordinal_features = profile.loc[
        profile["feature_type"] == "ordinal", "feature"
    ].tolist()

    continuous_features = profile.loc[
        profile["feature_type"] == "continuous", "feature"
    ].tolist()

    return (
        binary_features,
        ordinal_features,
        continuous_features,
    )


def build_preprocessor(scale_continuous=True):
    """
    Build preprocessing pipeline.

    Continuous:
        Median imputation
        Optional standard scaling

    Binary and ordinal:
        Most frequent imputation

    Constant features:
        Removed using VarianceThreshold.
    """

    (
        binary_features,
        ordinal_features,
        continuous_features,
    ) = get_feature_groups()

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    continuous_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_continuous:
        continuous_steps.append(
            ("scaler", StandardScaler())
        )

    continuous_pipeline = Pipeline(
        steps=continuous_steps
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("binary", binary_pipeline, binary_features),
            ("ordinal", ordinal_pipeline, ordinal_features),
            ("continuous", continuous_pipeline, continuous_features),
        ],
        remainder="drop",
    )

    preprocessor = Pipeline(
        steps=[
            ("column_transformer", column_transformer),
            ("variance_filter", VarianceThreshold()),
        ]
    )

    return preprocessor
