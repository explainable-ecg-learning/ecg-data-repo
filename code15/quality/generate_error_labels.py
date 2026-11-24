"""
Generate per-exam data-quality and error labels for all 345,779 CODE-15% exams.

Audits three full-cohort domains across exams.csv + available exams_part*.hdf5 files:
  1. File / HDF5 Availability & Longitudinal Integrity (err_file_*, err_tbl_*)
  2. Demographics & Clinical Outcome Metadata          (err_age_*, err_sex_*, err_meta_*)
  3. Multi-Label Clinical Consistency                  (err_lbl_*)

Usage:
    .venv/bin/python code15/quality/generate_error_labels.py

Output:
    code15/quality/error_labels.csv.gz
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

QUALITY_DIR = Path(__file__).resolve().parent
CODE15_DIR = QUALITY_DIR.parent
DATA_DIR = CODE15_DIR / "data"
OUT_CSV_GZ = QUALITY_DIR / "error_labels.csv.gz"

LABEL_COLS = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]


def main() -> None:
    t0 = time.time()
    print(f"Loading CODE-15% exams.csv and indexing HDF5 parts from {DATA_DIR} ...")
    df = pd.read_csv(DATA_DIR / "exams.csv")
    df.columns = df.columns.str.strip()

    # Index exam_ids actually present in extracted exams_part*.hdf5 files
    present_exam_ids: set[int] = set()
    for part_path in sorted(DATA_DIR.glob("exams_part*.hdf5")):
        with h5py.File(part_path, "r") as f:
            ids = np.array(f["exam_id"])
            present_exam_ids.update(int(x) for x in ids if int(x) > 0)

    out = pd.DataFrame({
        "exam_id": df["exam_id"].astype(np.int64),
        "patient_id": df["patient_id"].astype(np.int64),
        "trace_file": df["trace_file"].astype(str),
    })

    # 1. File availability & longitudinal consistency
    err_file_missing_hdf5_part = ~df["exam_id"].isin(present_exam_ids)

    sex_nunique = df.groupby("patient_id")["is_male"].nunique(dropna=True)
    sex_conflict_pids = set(sex_nunique[sex_nunique > 1].index)
    age_spread = df.groupby("patient_id")["age"].agg(lambda s: s.max() - s.min())
    age_jump_pids = set(age_spread[age_spread > 15].index)
    err_tbl_longitudinal_demo_conflict = df["patient_id"].isin(sex_conflict_pids | age_jump_pids)

    # 2. Demographics & survival/predicted-age metadata
    err_age_missing_or_invalid = df["age"].isna() | df["age"].le(0) | df["age"].gt(100)
    err_age_pediatric = df["age"].lt(18)
    err_age_nn_discrepancy_gt35 = (df["age"] - df["nn_predicted_age"]).abs().gt(35.0)
    err_meta_missing_survival = df["death"].isna() | df["timey"].isna()
    err_meta_negative_timey = df["timey"].lt(0)

    # 3. Label consistency
    any_abn = df[LABEL_COLS].astype(bool).any(axis=1)
    err_lbl_normal_with_abnormality = df["normal_ecg"].astype(bool) & any_abn
    err_lbl_af_with_sinus_rhythm = df["AF"].astype(bool) & (df["SB"].astype(bool) | df["ST"].astype(bool))
    err_lbl_sb_and_st_conflict = df["SB"].astype(bool) & df["ST"].astype(bool)
    err_lbl_rbbb_and_lbbb_conflict = df["RBBB"].astype(bool) & df["LBBB"].astype(bool)
    err_lbl_unlabeled_non_normal = (~df["normal_ecg"].astype(bool)) & (~any_abn)

    flag_map = {
        "err_file_missing_hdf5_part": err_file_missing_hdf5_part,
        "err_tbl_longitudinal_demo_conflict": err_tbl_longitudinal_demo_conflict,
        "err_age_missing_or_invalid": err_age_missing_or_invalid,
        "err_age_pediatric": err_age_pediatric,
        "err_age_nn_discrepancy_gt35": err_age_nn_discrepancy_gt35,
        "err_meta_missing_survival": err_meta_missing_survival,
        "err_meta_negative_timey": err_meta_negative_timey,
        "err_lbl_normal_with_abnormality": err_lbl_normal_with_abnormality,
        "err_lbl_af_with_sinus_rhythm": err_lbl_af_with_sinus_rhythm,
        "err_lbl_sb_and_st_conflict": err_lbl_sb_and_st_conflict,
        "err_lbl_rbbb_and_lbbb_conflict": err_lbl_rbbb_and_lbbb_conflict,
        "err_lbl_unlabeled_non_normal": err_lbl_unlabeled_non_normal,
    }

    flag_cols = list(flag_map.keys())
    for col, s in flag_map.items():
        out[col] = s.fillna(False).astype(np.int8)

    out["n_error_flags"] = out[flag_cols].sum(axis=1).astype(np.int8)

    severe_cols = [
        "err_file_missing_hdf5_part",
        "err_age_missing_or_invalid",
        "err_meta_negative_timey",
        "err_lbl_normal_with_abnormality",
        "err_lbl_sb_and_st_conflict",
    ]
    moderate_cols = [
        "err_tbl_longitudinal_demo_conflict",
        "err_age_nn_discrepancy_gt35",
        "err_lbl_af_with_sinus_rhythm",
        "err_lbl_rbbb_and_lbbb_conflict",
    ]

    is_severe = out[severe_cols].any(axis=1)
    is_moderate = ~is_severe & out[moderate_cols].any(axis=1)
    is_minor = ~is_severe & ~is_moderate & (out["n_error_flags"] > 0)

    out["severity_tier"] = np.select(
        [is_severe, is_moderate, is_minor],
        ["3_severe_error", "2_moderate_anomaly", "1_minor_warning"],
        default="0_clean",
    )
    out["recommended_action"] = np.select(
        [is_severe, is_moderate, is_minor],
        ["exclude", "review_or_downweight", "keep_with_cleaning"],
        default="keep",
    )

    short_names = [c[4:] for c in flag_cols]
    mat = out[flag_cols].to_numpy(dtype=bool)
    out["error_tags"] = [
        "|".join(short_names[j] for j in np.flatnonzero(row_mask))
        for row_mask in mat
    ]

    out.to_csv(OUT_CSV_GZ, index=False, compression="gzip")
    elapsed = time.time() - t0
    print(f"\nSaved {len(out):,} rows × {len(out.columns)} cols to {OUT_CSV_GZ} in {elapsed:.1f}s ({OUT_CSV_GZ.stat().st_size / (1024*1024):.2f} MB)")
    print("\n=== Severity Tier Breakdown ===")
    print(out["severity_tier"].value_counts().sort_index())
    print("\n=== Individual Error Flag Counts ===")
    for c in flag_cols:
        cnt = int(out[c].sum())
        print(f"  {c:<36}: {cnt:7,d} ({cnt / len(out) * 100:5.2f}%)")


if __name__ == "__main__":
    main()
