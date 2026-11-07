
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "X_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "X_test.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

binary_features = """
SEX IBS_NASL SIM_GIPERT
nr11 nr01 nr02 nr03 nr04 nr07 nr08
np01 np04 np05 np07 np08 np09 np10
endocr_01 endocr_02 endocr_03
zab_leg_01 zab_leg_02 zab_leg_03 zab_leg_04 zab_leg_06
O_L_POST K_SH_POST MP_TP_POST SVT_POST GT_POST FIB_G_POST
ritm_ecg_p_01 ritm_ecg_p_02 ritm_ecg_p_04
ritm_ecg_p_06 ritm_ecg_p_07 ritm_ecg_p_08
n_r_ecg_p_01 n_r_ecg_p_02 n_r_ecg_p_03
n_r_ecg_p_04 n_r_ecg_p_05 n_r_ecg_p_06
n_r_ecg_p_08 n_r_ecg_p_09 n_r_ecg_p_10
n_p_ecg_p_01 n_p_ecg_p_03 n_p_ecg_p_04
n_p_ecg_p_05 n_p_ecg_p_06 n_p_ecg_p_07
n_p_ecg_p_08 n_p_ecg_p_09 n_p_ecg_p_10
n_p_ecg_p_11 n_p_ecg_p_12
fibr_ter_01 fibr_ter_02 fibr_ter_03 fibr_ter_05
fibr_ter_06 fibr_ter_07 fibr_ter_08
GIPO_K GIPER_Na
NA_KB NOT_NA_KB LID_KB NITR_S
LID_S_n B_BLOK_S_n ANT_CA_S_n GEPAR_S_n
ASP_S_n TIKL_S_n TRENT_S_n IM_PG_P
""".split()

ordinal_features = """
INF_ANAM STENOK_AN FK_STENOK IBS_POST GB ZSN_A
ant_im lat_im inf_im post_im TIME_B_S
""".split()

continuous_features = """
AGE DLIT_AG
S_AD_KBRIG D_AD_KBRIG S_AD_ORIT D_AD_ORIT
K_BLOOD Na_BLOOD ALT_BLOOD AST_BLOOD
KFK_BLOOD L_BLOOD ROE
""".split()

feature_groups = {
    "binary": set(binary_features),
    "ordinal": set(ordinal_features),
    "continuous": set(continuous_features)
}

all_mapped_features = (
    feature_groups["binary"]
    | feature_groups["ordinal"]
    | feature_groups["continuous"]
)

all_grouped_features = (
    feature_groups["binary"]
    | feature_groups["ordinal"]
    | feature_groups["continuous"]
)

overlapping_features = (
    (feature_groups["binary"] & feature_groups["ordinal"])
    | (feature_groups["binary"] & feature_groups["continuous"])
    | (feature_groups["ordinal"] & feature_groups["continuous"])
)

if overlapping_features:
    raise ValueError(
        f"Features assigned to multiple groups: {sorted(overlapping_features)}"
    )

X_train = pd.read_csv(TRAIN_PATH)
X_test = pd.read_csv(TEST_PATH)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and Test feature columns do not match.")

undefined_features = sorted(set(X_train.columns) - all_mapped_features)
unknown_features = sorted(all_mapped_features - set(X_train.columns))

if undefined_features or unknown_features:
    raise ValueError(
        f"Feature mapping mismatch. Undefined: {undefined_features}, "
        f"Unknown: {unknown_features}"
    )

if len(all_mapped_features) != X_train.shape[1]:
    raise ValueError("The number of mapped features does not match the dataset.")

feature_type_map = {
    feature: feature_type
    for feature_type, features in feature_groups.items()
    for feature in features
}

records = []

for feature in X_train.columns:
    train_values = sorted(X_train[feature].dropna().unique().tolist())
    test_values = sorted(X_test[feature].dropna().unique().tolist())

    feature_type = feature_type_map[feature]

    invalid_train_values = []
    invalid_test_values = []

    if feature_type == "binary":
        invalid_train_values = [
            value for value in train_values if value not in [0, 1]
        ]

        invalid_test_values = [
            value for value in test_values if value not in [0, 1]
        ]

    records.append({
        "feature": feature,
        "feature_type": feature_type,
        "train_missing_count": int(X_train[feature].isna().sum()),
        "train_missing_percent": round(
            X_train[feature].isna().mean() * 100, 2
        ),
        "test_missing_count": int(X_test[feature].isna().sum()),
        "test_missing_percent": round(
            X_test[feature].isna().mean() * 100, 2
        ),
        "train_unique_values": len(train_values),
        "train_observed_values": str(train_values),
        "test_unique_values": len(test_values),
        "test_observed_values": str(test_values),
        "invalid_train_values": str(invalid_train_values),
        "invalid_test_values": str(invalid_test_values),
        "constant_in_train": len(train_values) <= 1
    })

profile = pd.DataFrame(records)

type_summary = (
    profile["feature_type"]
    .value_counts()
    .rename_axis("feature_type")
    .reset_index(name="feature_count")
)

profile_path = OUTPUT_DIR / "feature_type_profile.csv"
summary_path = OUTPUT_DIR / "feature_type_summary.csv"

profile.to_csv(profile_path, index=False)
type_summary.to_csv(summary_path, index=False)

print("=" * 70)
print("FEATURE TYPE ANALYSIS")
print("=" * 70)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Total mapped features: {len(feature_type_map)}")

print("\nFEATURE TYPE SUMMARY")
print(type_summary.to_string(index=False))

print("\nBINARY VALIDATION")

invalid_binary = profile[
    (profile["invalid_train_values"] != "[]")
    | (profile["invalid_test_values"] != "[]")
]

if invalid_binary.empty:
    print("All binary features contain only 0 and 1 in Train and Test.")
else:
    print(
        invalid_binary[
            [
                "feature",
                "train_observed_values",
                "test_observed_values",
                "invalid_train_values",
                "invalid_test_values"
            ]
        ].to_string(index=False)
    )

print("\nCONSTANT FEATURES IN TRAIN")

constant_features = profile[profile["constant_in_train"]]

if constant_features.empty:
    print("No constant features found.")
else:
    print(
        constant_features[
            ["feature", "feature_type", "train_observed_values"]
        ].to_string(index=False)
    )

print("\nFEATURES WITH HIGH MISSINGNESS IN TRAIN")

high_missing = profile[profile["train_missing_percent"] >= 40]

if high_missing.empty:
    print("No features with missingness >= 40%.")
else:
    print(
        high_missing[
            ["feature", "feature_type", "train_missing_percent"]
        ].to_string(index=False)
    )

print("\nFEATURE PROFILE")
print(profile.to_string(index=False))

print("\nSAVED FILES")
print(profile_path.relative_to(ROOT))
print(summary_path.relative_to(ROOT))
