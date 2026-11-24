"""
Generate per-study data-quality and error labels for all 800,035 MIMIC-IV-ECG studies.

Audits three full-cohort domains in vectorized pandas/numpy (~15s for 800,035 rows):
  1. Table, Timestamp & Hardware Provenance (err_tbl_*, err_hw_*)
  2. Machine Measurement Sentinels & Ranges (err_meas_* — 29999, int16/uint16 overflows, impossible intervals)
  3. Machine Report Quality & Rhythm Checks (err_rep_* — lead reversals, poor data/artifact, pacemaker, HR mismatch)

Usage:
    .venv/bin/python mimic_iv_ecg/quality/generate_error_labels.py

Output:
    mimic_iv_ecg/quality/error_labels.csv.gz
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

QUALITY_DIR = Path(__file__).resolve().parent
MIMIC_DIR = QUALITY_DIR.parent
DATA_DIR = MIMIC_DIR / "data"
OUT_CSV_GZ = QUALITY_DIR / "error_labels.csv.gz"


def main() -> None:
    t0 = time.time()
    print(f"Loading MIMIC-IV-ECG tables from {DATA_DIR} ...")
    rl = pd.read_csv(DATA_DIR / "record_list.csv")
    mm = pd.read_csv(DATA_DIR / "machine_measurements.csv", low_memory=False)
    wl = pd.read_csv(DATA_DIR / "waveform_note_links.csv", usecols=["subject_id", "study_id"])

    rl.columns = rl.columns.str.strip().str.lower()
    mm.columns = mm.columns.str.strip().str.lower()
    wl.columns = wl.columns.str.strip().str.lower()

    # Align mm strictly on record_list study_id order
    mm = mm.set_index("study_id").reindex(rl["study_id"]).reset_index()

    out = pd.DataFrame({
        "study_id": rl["study_id"].astype(np.int64),
        "subject_id": rl["subject_id"].astype(np.int64),
    })

    # ── 1. Table, Timestamp & Hardware Provenance ─────────────────────────
    wl_counts = wl["study_id"].value_counts()
    wl_study_set = set(wl_counts.index)
    wl_dup_set = set(wl_counts[wl_counts > 1].index)

    err_tbl_duplicate_note_link = rl["study_id"].isin(wl_dup_set)
    err_tbl_unlinked_note = ~rl["study_id"].isin(wl_study_set)
    err_tbl_duplicate_subject_timestamp = rl.duplicated(subset=["subject_id", "ecg_time"], keep=False)

    filt_s = mm["filtering"].fillna("").astype(str).str.strip()
    err_hw_filtering_unspecified = filt_s.isin(["", "<not specified>"])
    err_hw_50hz_notch_in_us = filt_s.str.contains("50 Hz", case=False, regex=False)

    # ── 2. Machine Measurement Sentinels & Physiological Ranges ───────────
    core_cols = ["rr_interval", "qrs_onset", "qrs_end", "t_end", "qrs_axis", "t_axis"]
    pwave_cols = ["p_onset", "p_end", "p_axis"]

    err_meas_sentinel_29999_core = (mm[core_cols] == 29999).any(axis=1)
    err_meas_sentinel_29999_pwave = (mm[pwave_cols] == 29999).any(axis=1)

    # Replace 29999 with NaN to inspect remaining overflow / impossible values
    m = mm[core_cols + pwave_cols].replace(29999, np.nan)

    err_meas_int16_overflow_or_zero = (
        m["rr_interval"].le(100) | m["rr_interval"].gt(4000)
        | m["p_onset"].gt(5000) | m["p_end"].gt(5000)
        | m["qrs_onset"].gt(5000) | m["qrs_end"].gt(5000) | m["t_end"].gt(5000)
        | m["p_axis"].lt(-180) | m["p_axis"].gt(360)
        | m["qrs_axis"].lt(-180) | m["qrs_axis"].gt(360)
        | m["t_axis"].lt(-180) | m["t_axis"].gt(360)
    )

    qrs_dur = m["qrs_end"] - m["qrs_onset"]
    pr_int = m["qrs_onset"] - m["p_onset"]
    qt_int = m["t_end"] - m["qrs_onset"]

    err_meas_impossible_intervals = (
        (qrs_dur.notna() & ((qrs_dur < 30) | (qrs_dur > 300)))
        | (pr_int.notna() & ((pr_int < 50) | (pr_int > 500)))
        | (qt_int.notna() & ((qt_int < 180) | (qt_int > 750)))
    )

    # ── 3. Machine Report Quality Warnings & Clinical Consistency ─────────
    rep_cols = [f"report_{i}" for i in range(18) if f"report_{i}" in mm.columns]
    err_rep_empty_all = mm[rep_cols].isna().all(axis=1)

    rep_text = mm[rep_cols].fillna("").agg(" | ".join, axis=1).str.lower()

    err_rep_lead_reversal = rep_text.str.contains(
        r"lead reversal|arm lead|limb lead|revers(?:ed|al).*lead|suspect.*lead",
        regex=True,
    )
    err_rep_poor_quality_or_artifact = rep_text.str.contains(
        r"poor data|technical quality|artifact|baseline wander|ac interference|muscle tremor|uninterpretable|cannot be|invalid ecg|noise",
        regex=True,
    )
    err_rep_pacemaker_or_paced = rep_text.str.contains(r"pacemaker|paced", regex=True)
    err_rep_pediatric = rep_text.str.contains(r"pediatric", regex=True)

    # Machine RR vs reported sinus bradycardia / sinus tachycardia
    valid_rr = m["rr_interval"].where(m["rr_interval"].between(200, 3000))
    hr_bpm = 60000.0 / valid_rr
    is_sbrad_rep = rep_text.str.contains("sinus bradycardia", regex=False)
    is_stach_rep = rep_text.str.contains("sinus tachycardia", regex=False)
    err_rep_rhythm_hr_mismatch = (
        (is_sbrad_rep & hr_bpm.gt(68.0))
        | (is_stach_rep & hr_bpm.lt(92.0))
    )

    flag_map = {
        "err_tbl_duplicate_note_link": err_tbl_duplicate_note_link,
        "err_tbl_unlinked_note": err_tbl_unlinked_note,
        "err_tbl_duplicate_subject_timestamp": err_tbl_duplicate_subject_timestamp,
        "err_hw_filtering_unspecified": err_hw_filtering_unspecified,
        "err_hw_50hz_notch_in_us": err_hw_50hz_notch_in_us,
        "err_meas_sentinel_29999_core": err_meas_sentinel_29999_core,
        "err_meas_sentinel_29999_pwave": err_meas_sentinel_29999_pwave,
        "err_meas_int16_overflow_or_zero": err_meas_int16_overflow_or_zero,
        "err_meas_impossible_intervals": err_meas_impossible_intervals,
        "err_rep_empty_all": err_rep_empty_all,
        "err_rep_lead_reversal": err_rep_lead_reversal,
        "err_rep_poor_quality_or_artifact": err_rep_poor_quality_or_artifact,
        "err_rep_pacemaker_or_paced": err_rep_pacemaker_or_paced,
        "err_rep_pediatric": err_rep_pediatric,
        "err_rep_rhythm_hr_mismatch": err_rep_rhythm_hr_mismatch,
    }

    flag_cols = list(flag_map.keys())
    for col, s in flag_map.items():
        out[col] = s.fillna(False).astype(np.int8)

    out["n_error_flags"] = out[flag_cols].sum(axis=1).astype(np.int8)

    severe_cols = [
        "err_rep_lead_reversal",
        "err_rep_poor_quality_or_artifact",
        "err_rep_empty_all",
        "err_meas_sentinel_29999_core",
        "err_meas_int16_overflow_or_zero",
        "err_meas_impossible_intervals",
    ]
    moderate_cols = [
        "err_tbl_duplicate_subject_timestamp",
        "err_hw_50hz_notch_in_us",
        "err_rep_rhythm_hr_mismatch",
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

    # Compact pipe-separated tags
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
