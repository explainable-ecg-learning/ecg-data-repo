"""
Generate per-record data quality and error labels for all 21,799 PTB-XL ECGs.

Combines four audit domains discovered during exploratory data analysis:
  1. Metadata & Demographic Errors  (err_age_*, err_anthro_*, err_longitudinal_*)
  2. Clinical & SCP Label Errors    (err_scp_*, err_infarction_*, err_rhythm_*)
  3. Annotated Signal Artifacts     (err_art_* + n_annotated_noisy_leads)
  4. Waveform & Delineation Defects (err_sig_*, err_seg_*) across all 21,799 records

Usage:
    .venv/bin/python ptbxl/quality/generate_error_labels.py

Output:
    ptbxl/quality/error_labels.csv
"""

from __future__ import annotations

import ast
import io
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sp_signal

QUALITY_DIR = Path(__file__).resolve().parent
PTBXL_DIR = QUALITY_DIR.parent
REPO_ROOT = PTBXL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptbxl.dataset import PTBXLDataset

DATA_DIR = PTBXL_DIR / "data"
SEG_ZIP = DATA_DIR / "ptbxl_segmentations_8bit.zip"
OUT_CSV = QUALITY_DIR / "error_labels.csv"

LEADS = PTBXLDataset.LEADS
LEAD_IDX = {ld: i for i, ld in enumerate(LEADS)}
LOG_24 = float(np.log(24.0))


def parse_lead_mask(val) -> np.ndarray:
    """Parse a PTB-XL free-text lead range annotation into a (12,) bool mask."""
    mask = np.zeros(12, dtype=bool)
    if pd.isna(val):
        return mask
    raw = str(val).upper().replace(";", ",")
    tokens = [t.strip(" ,.") for t in re.split(r"[,\s]+", raw) if t.strip(" ,.")]
    for tok in tokens:
        if tok in ("ALLES", "ALLE", "ALL"):
            mask[:] = True
        elif tok in LEAD_IDX:
            mask[LEAD_IDX[tok]] = True
        elif "-" in tok:
            p0, p1 = [p.strip() for p in tok.split("-", 1)]
            if p0.startswith("V") and p1.isdigit():
                p1 = "V" + p1
            if p0 in LEAD_IDX and p1 in LEAD_IDX:
                i0, i1 = sorted((LEAD_IDX[p0], LEAD_IDX[p1]))
                mask[i0 : i1 + 1] = True
    return mask


def build_seg_member_map(zip_path: Path) -> dict[int, str]:
    """Index ecg_id -> zip member filename from Zenodo 7610236 archive."""
    mapping: dict[int, str] = {}
    if not zip_path.exists():
        return mapping
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.is_dir():
                m = re.match(r"^0*(\d+)(?:\..+)?$", Path(info.filename).name)
                if m:
                    mapping[int(m.group(1))] = info.filename
    return mapping


def main() -> None:
    t0 = time.time()
    print(f"Loading PTB-XL metadata from {DATA_DIR} ...")
    df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_dict"] = df["scp_codes"].apply(ast.literal_eval)
    df["recording_date_dt"] = pd.to_datetime(df["recording_date"], errors="coerce")

    scp_df = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
    diag_scp = scp_df[scp_df["diagnostic"] == 1.0]
    diag_codes = set(diag_scp.index)
    mi_codes = set(diag_scp.index[diag_scp["diagnostic_class"] == "MI"])

    # Subclass -> expected parent superclass map
    sub_to_super = {
        PTBXLDataset._normalize_subclass(row["diagnostic_subclass"]): row["diagnostic_class"]
        for _, row in diag_scp.iterrows()
    }
    sub_to_super["NORM"] = "NORM"

    ds_helper = PTBXLDataset.__new__(PTBXLDataset)

    # ── 1. Metadata & Demographic Error Flags ─────────────────────────────
    err_age_sentinel_300 = df["age"].gt(100)
    err_age_pediatric = df["age"].lt(18)

    bmi = df["weight"] / ((df["height"] / 100.0) ** 2)
    is_adult_or_teen = df["age"].ge(16) | df["age"].isna()
    err_anthro_typo = (
        (is_adult_or_teen & (df["height"].lt(120) | df["height"].gt(215) | df["weight"].lt(28) | df["weight"].gt(230)))
        | (~is_adult_or_teen & (df["height"].lt(40) | df["height"].gt(195) | df["weight"].lt(2.5) | bmi.lt(10) | bmi.gt(45)))
        | bmi.lt(12.0)
        | bmi.gt(65.0)
    )

    # Longitudinal within-patient contradictions (sex flip, age drop >1 yr, height diff >5 cm, or >25 kg weight jump within 90 days)
    df_sorted = df.sort_values(["patient_id", "recording_date_dt"])
    sex_flip_pids = set(
        df_sorted.groupby("patient_id")["sex"].nunique(dropna=True).loc[lambda s: s > 1].index
    )
    valid_age_sorted = df_sorted[df_sorted["age"] <= 100]
    age_drop_pids = set(
        valid_age_sorted.loc[valid_age_sorted.groupby("patient_id")["age"].diff() < -1.0, "patient_id"]
    )
    adult_sorted = df_sorted[df_sorted["age"] >= 20].copy()
    h_jump_pids = set(
        adult_sorted.groupby("patient_id")["height"]
        .agg(lambda s: (s.max() - s.min()) if s.notna().sum() >= 2 else 0.0)
        .loc[lambda s: s >= 6.0]
        .index
    )
    adult_sorted["dt_days"] = adult_sorted.groupby("patient_id")["recording_date_dt"].diff().dt.days
    adult_sorted["dw_abs"] = adult_sorted.groupby("patient_id")["weight"].diff().abs()
    w_jump_pids = set(
        adult_sorted.loc[(adult_sorted["dt_days"] <= 90) & (adult_sorted["dw_abs"] >= 25.0), "patient_id"]
    )
    conflict_pids = sex_flip_pids | age_drop_pids | h_jump_pids | w_jump_pids
    err_longitudinal_demo_conflict = df["patient_id"].isin(conflict_pids)

    # ── 2. Clinical & SCP Label Error Flags ───────────────────────────────
    def max_diag_w(d: dict) -> float:
        vals = [float(w) for c, w in d.items() if c in diag_codes]
        return max(vals) if vals else -1.0

    max_dw = df["scp_dict"].apply(max_diag_w)
    err_scp_zero_diagnostic = max_dw.eq(-1.0)
    err_scp_low_confidence = max_dw.ge(0.0) & max_dw.le(35.0)

    def has_norm_and_path(d: dict) -> bool:
        supers = {diag_scp.loc[c, "diagnostic_class"] for c in d if c in diag_codes}
        return "NORM" in supers and len(supers) > 1

    err_scp_norm_pathology_conflict = df["scp_dict"].apply(has_norm_and_path)

    picked_sub = df["scp_dict"].apply(lambda d: ds_helper._pick_subclass(d, diag_scp)[0])
    agg_super = df["scp_dict"].apply(lambda d: ds_helper._aggregate_class(d, diag_scp))
    err_scp_hierarchy_mismatch = agg_super != picked_sub.map(sub_to_super)

    err_scp_unvalidated_autogen = ~df["validated_by_human"].astype(bool)

    has_inf_stad = df["infarction_stadium1"].notna() | df["infarction_stadium2"].notna()
    has_mi_scp = df["scp_dict"].apply(lambda d: any(c in mi_codes for c in d))
    err_infarction_stadium_no_mi = has_inf_stad & ~has_mi_scp

    # ── 3. Annotated Signal Artifacts ─────────────────────────────────────
    err_art_electrodes_problem = df["electrodes_problems"].notna()
    err_art_burst_noise = df["burst_noise"].notna()
    err_art_static_noise = df["static_noise"].notna()
    err_art_baseline_drift = df["baseline_drift"].notna()
    err_art_pacemaker = df["pacemaker"].notna()

    lead_union = np.zeros((len(df), 12), dtype=bool)
    for col in ("electrodes_problems", "burst_noise", "static_noise", "baseline_drift"):
        lead_union |= np.vstack([parse_lead_mask(v) for v in df[col]])
    n_annotated_noisy_leads = pd.Series(lead_union.sum(axis=1), index=df.index, dtype=int)

    # ── 4. Full 21,799 Waveform & Soft-Segmentation Scan ──────────────────
    seg_map = build_seg_member_map(SEG_ZIP)
    has_segs = len(seg_map) > 0
    print(f"Scanning all {len(df):,} 100-Hz waveforms (and {len(seg_map):,} soft segmentations) across 16 threads ...")

    tls = threading.local()
    b_qrs, a_qrs = sp_signal.butter(3, [5.0 / 50.0, 20.0 / 50.0], btype="bandpass")
    rel_paths = df["filename_lr"].to_dict()

    def scan_one(ecg_id: int) -> tuple:
        sig, _ = wfdb.rdsamp(str(DATA_DIR / rel_paths[ecg_id]))
        has_nan_inf = bool(np.isnan(sig).any() or np.isinf(sig).any())
        if has_nan_inf:
            sig = np.nan_to_num(sig)

        lead_std = sig.std(axis=0)
        max_abs = float(np.abs(sig).max())
        max_dc = float(np.abs(sig.mean(axis=0)).max())

        # Longest constant run across leads
        zero_diff = np.diff(sig, axis=0) == 0.0
        max_flat_samples = 0
        for j in range(12):
            z = zero_diff[:, j]
            if z.any():
                padded = np.concatenate([[False], z, [False]])
                starts = np.flatnonzero(padded[1:] & ~padded[:-1])
                ends = np.flatnonzero(~padded[1:] & padded[:-1])
                run = int((ends - starts).max())
                if run > max_flat_samples:
                    max_flat_samples = run

        flatline = bool((lead_std.min() < 0.005) or (max_flat_samples >= 50))
        extreme_amp = bool(max_abs > 10.0)
        large_dc = bool(max_dc > 0.5)

        einth_rmse = float(np.sqrt(np.mean((sig[:, 1] - (sig[:, 0] + sig[:, 2])) ** 2)))
        gold_rmse = float(np.sqrt(np.mean((sig[:, 3] + sig[:, 4] + sig[:, 5]) ** 2)))
        physics_bad = bool((einth_rmse > 0.05) or (gold_rmse > 0.05))

        # Soft segmentation uncertainty, cross-lead R-peak sync & HR estimate
        high_ent = False
        lead_desync = False
        hr_bpm = np.nan

        if has_segs and ecg_id in seg_map:
            zf = getattr(tls, "zf", None)
            if zf is None:
                zf = zipfile.ZipFile(SEG_ZIP, "r")
                tls.zf = zf
            raw = zf.read(seg_map[ecg_id])
            u8 = np.asarray(np.load(io.BytesIO(raw), allow_pickle=True), dtype=np.float32)
            probs = u8 / np.maximum(u8.sum(axis=-1, keepdims=True), 1.0)

            ent = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=-1) / LOG_24
            max_lead_ent = float(ent.mean(axis=0).max())
            high_ent = bool(max_lead_ent > 0.45)

            # Cross-lead QRS-complex synchrony (channels 6..16 = QRS-Onset..J-point, invariant to aVR polarity!)
            qrs_prob = probs[:, :, 6:17].sum(axis=-1)
            med_qrs = np.median(qrs_prob, axis=1)
            med_centered = med_qrs - med_qrs.mean()
            med_norm = float(np.linalg.norm(med_centered)) + 1e-8
            min_corr = 1.0
            for j in range(12):
                ql_c = qrs_prob[:, j] - qrs_prob[:, j].mean()
                c_val = float(np.dot(ql_c, med_centered) / ((float(np.linalg.norm(ql_c)) + 1e-8) * med_norm))
                if c_val < min_corr:
                    min_corr = c_val
            lead_desync = bool(min_corr < 0.50)

            # Global QRS-peak HR estimate (using R-Peak + S-Peak for inverted QRS complexes)
            beat_trace = probs[:, :, 10].mean(axis=1) + probs[:, :, 12].mean(axis=1)
            peaks, _ = sp_signal.find_peaks(beat_trace, distance=25, height=0.15)
            if len(peaks) >= 4:
                hr_bpm = float(60.0 / (np.median(np.diff(peaks)) / 100.0))

        if np.isnan(hr_bpm):
            s_ii = sig[:, 1] - np.median(sig[:, 1])
            e_ii = sp_signal.filtfilt(b_qrs, a_qrs, s_ii) ** 2
            pks, _ = sp_signal.find_peaks(e_ii, distance=25, height=max(float(np.percentile(e_ii, 85)) * 1.4, 1e-4))
            if len(pks) >= 4:
                hr_bpm = float(60.0 / (np.median(np.diff(pks)) / 100.0))

        return (
            ecg_id,
            has_nan_inf,
            flatline,
            extreme_amp,
            large_dc,
            physics_bad,
            high_ent,
            lead_desync,
            hr_bpm,
        )

    eids = df.index.astype(int).tolist()
    with ThreadPoolExecutor(max_workers=16) as pool:
        scan_rows = list(pool.map(scan_one, eids, chunksize=64))

    scan_df = pd.DataFrame(
        scan_rows,
        columns=[
            "ecg_id",
            "err_sig_nan_or_inf",
            "err_sig_flatline_lead",
            "err_sig_extreme_amplitude",
            "err_sig_large_dc_offset",
            "err_sig_lead_physics_violation",
            "err_seg_high_uncertainty",
            "err_seg_lead_desync",
            "est_hr_bpm",
        ],
    ).set_index("ecg_id")

    # Rhythm vs waveform HR contradiction (SBRAD > 68 BPM or STACH < 92 BPM)
    has_sbrad = df["scp_dict"].apply(lambda d: "SBRAD" in d)
    has_stach = df["scp_dict"].apply(lambda d: "STACH" in d)
    err_rhythm_hr_mismatch = (
        (has_sbrad & scan_df["est_hr_bpm"].gt(68.0))
        | (has_stach & scan_df["est_hr_bpm"].lt(92.0))
    )

    any_meta_noise = err_art_electrodes_problem | err_art_burst_noise | err_art_static_noise
    err_sig_unannotated_defect = (
        ~any_meta_noise
        & (
            scan_df["err_sig_flatline_lead"]
            | scan_df["err_sig_extreme_amplitude"]
            | scan_df["err_sig_lead_physics_violation"]
        )
    )

    # ── Assemble Output DataFrame ─────────────────────────────────────────
    out = pd.DataFrame(index=df.index)
    out["strat_fold"] = df["strat_fold"].astype(int)

    flag_series = {
        # Metadata
        "err_age_sentinel_300": err_age_sentinel_300,
        "err_age_pediatric": err_age_pediatric,
        "err_anthro_typo": err_anthro_typo,
        "err_longitudinal_demo_conflict": err_longitudinal_demo_conflict,
        # Clinical / SCP labels
        "err_scp_zero_diagnostic": err_scp_zero_diagnostic,
        "err_scp_low_confidence": err_scp_low_confidence,
        "err_scp_norm_pathology_conflict": err_scp_norm_pathology_conflict,
        "err_scp_hierarchy_mismatch": err_scp_hierarchy_mismatch,
        "err_scp_unvalidated_autogen": err_scp_unvalidated_autogen,
        "err_infarction_stadium_no_mi": err_infarction_stadium_no_mi,
        "err_rhythm_hr_mismatch": err_rhythm_hr_mismatch,
        # Annotated artifacts
        "err_art_electrodes_problem": err_art_electrodes_problem,
        "err_art_burst_noise": err_art_burst_noise,
        "err_art_static_noise": err_art_static_noise,
        "err_art_baseline_drift": err_art_baseline_drift,
        "err_art_pacemaker": err_art_pacemaker,
        # Waveform & delineation defects
        "err_sig_nan_or_inf": scan_df["err_sig_nan_or_inf"],
        "err_sig_flatline_lead": scan_df["err_sig_flatline_lead"],
        "err_sig_extreme_amplitude": scan_df["err_sig_extreme_amplitude"],
        "err_sig_large_dc_offset": scan_df["err_sig_large_dc_offset"],
        "err_sig_lead_physics_violation": scan_df["err_sig_lead_physics_violation"],
        "err_sig_unannotated_defect": err_sig_unannotated_defect,
        "err_seg_high_uncertainty": scan_df["err_seg_high_uncertainty"],
        "err_seg_lead_desync": scan_df["err_seg_lead_desync"],
    }

    for col_name, s in flag_series.items():
        out[col_name] = s.astype(np.int8)

    flag_cols = list(flag_series.keys())
    out["n_annotated_noisy_leads"] = n_annotated_noisy_leads.astype(np.int8)
    out["n_error_flags"] = out[flag_cols].sum(axis=1).astype(np.int8)

    # Severity tier & recommended action
    severe_cols = [
        "err_scp_zero_diagnostic",
        "err_scp_norm_pathology_conflict",
        "err_art_electrodes_problem",
        "err_sig_nan_or_inf",
        "err_sig_flatline_lead",
        "err_sig_extreme_amplitude",
        "err_sig_lead_physics_violation",
    ]
    moderate_cols = [
        "err_age_sentinel_300",
        "err_anthro_typo",
        "err_longitudinal_demo_conflict",
        "err_scp_low_confidence",
        "err_infarction_stadium_no_mi",
        "err_rhythm_hr_mismatch",
        "err_art_burst_noise",
        "err_seg_high_uncertainty",
        "err_seg_lead_desync",
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

    # Concise pipe-delimited active error tags (strip leading 'err_')
    short_names = [c[4:] for c in flag_cols]
    mat = out[flag_cols].to_numpy(dtype=bool)
    out["error_tags"] = [
        "|".join(short_names[j] for j in np.flatnonzero(row_mask))
        for row_mask in mat
    ]

    out.to_csv(OUT_CSV)
    elapsed = time.time() - t0
    print(f"\nSaved {len(out):,} rows × {len(out.columns)} cols to {OUT_CSV} in {elapsed:.1f}s ({OUT_CSV.stat().st_size / 1024:.1f} KB)")
    print("\n=== Severity Tier Breakdown ===")
    print(out["severity_tier"].value_counts().sort_index())
    print("\n=== Individual Error Flag Counts ===")
    for c in flag_cols:
        cnt = int(out[c].sum())
        print(f"  {c:<34}: {cnt:6,d} ({cnt / len(out) * 100:5.2f}%)")


if __name__ == "__main__":
    main()
