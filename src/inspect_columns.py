
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/MI.data")
OUTPUT_DIR = Path("outputs/metrics")

feature_names = """
AGE SEX INF_ANAM STENOK_AN FK_STENOK IBS_POST IBS_NASL GB SIM_GIPERT DLIT_AG ZSN_A
nr11 nr01 nr02 nr03 nr04 nr07 nr08
np01 np04 np05 np07 np08 np09 np10
endocr_01 endocr_02 endocr_03
zab_leg_01 zab_leg_02 zab_leg_03 zab_leg_04 zab_leg_06
S_AD_KBRIG D_AD_KBRIG S_AD_ORIT D_AD_ORIT
O_L_POST K_SH_POST MP_TP_POST SVT_POST GT_POST FIB_G_POST
ant_im lat_im inf_im post_im IM_PG_P
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
GIPO_K K_BLOOD GIPER_Na Na_BLOOD ALT_BLOOD
AST_BLOOD KFK_BLOOD L_BLOOD ROE TIME_B_S
R_AB_1_n R_AB_2_n R_AB_3_n
NA_KB NOT_NA_KB LID_KB NITR_S
NA_R_1_n NA_R_2_n NA_R_3_n
NOT_NA_1_n NOT_NA_2_n NOT_NA_3_n
LID_S_n B_BLOK_S_n ANT_CA_S_n GEPAR_S_n
ASP_S_n TIKL_S_n TRENT_S_n
""".split()

target_names = [
    "FIBR_PREDS",
    "PREDS_TAH",
    "JELUD_TAH",
    "FIBR_JELUD",
    "A_V_BLOK",
    "OTEK_LANC",
    "RAZRIV",
    "DRESSLER",
    "ZSN",
    "REC_IM",
    "P_IM_STEN",
    "LET_IS"
]

column_names = ["ID"] + feature_names + target_names

df = pd.read_csv(
    DATA_PATH,
    header=None,
    sep=",",
    na_values=["?"]
)

if len(column_names) != df.shape[1]:
    raise ValueError(
        f"Expected 124 column names, got {len(column_names)}"
    )

df.columns = column_names

print(f"Dataset shape: {df.shape}")
print(f"Feature count: {len(feature_names)}")
print(f"Target count: {len(target_names)}")

print("\nCOLUMN MAPPING")

mapping = pd.DataFrame({
    "file_column": range(1, len(column_names) + 1),
    "pandas_index": range(len(column_names)),
    "column_name": column_names,
    "missing_count": df.isna().sum().values,
    "missing_percent": (
        df.isna().mean().values * 100
    ).round(2)
})

print(mapping.to_string(index=False))

forbidden_file_columns = [93, 94, 95, 100, 101, 102, 103, 104, 105]

forbidden_features = [
    column_names[column - 1]
    for column in forbidden_file_columns
]

print("\nFORBIDDEN FEATURES")
print(forbidden_features)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mapping.to_csv(
    OUTPUT_DIR / "column_mapping.csv",
    index=False
)

print("\nSaved: outputs/metrics/column_mapping.csv")
