"""
Generate per-record metadata summary and data-quality error labels for all
10,344 Georgia 12-Lead ECG records (PhysioNet / CinC 2020 Challenge).

Audits four domains across all records in georgia_ecg/data/RECORDS:
  1. File & Header Integrity     (err_file_*, err_hdr_*)
  2. Demographics & Metadata     (err_age_*, err_sex_*)
  3. SNOMED-CT Label Quality     (err_dx_*, err_rhythm_*)
  4. Quantitative 500 Hz Signal  (err_sig_* — NaN, flatline, >10mV, DC, Einthoven/Goldberger, 60Hz/wander)

Usage:
    .venv/bin/python georgia_ecg/quality/generate_error_labels.py

Output:
    georgia_ecg/quality/error_labels.csv
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sp_signal

QUALITY_DIR = Path(__file__).resolve().parent
GEORGIA_DIR = QUALITY_DIR.parent
DATA_DIR = GEORGIA_DIR / "data"
OUT_CSV = QUALITY_DIR / "error_labels.csv"

EXPECTED_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

# SNOMED-CT codes for rhythm consistency checks
SNOMED_NSR = "426783006"
SNOMED_SB = {"426177001", "426627000"}       # Sinus bradycardia / Bradycardia
SNOMED_STACH = {"427084000"}                 # Sinus tachycardia
SNOMED_NSR_CONFLICTS = {
    "164889003",  # AF (Atrial fibrillation)
    "164890007",  # AFL (Atrial flutter)
    "10370003",   # PR (Pacing rhythm)
    "426761007",  # SVT (Supraventricular tachycardia)
    "426177001",  # SB
    "427084000",  # STach
}


def load_snomed_maps() -> tuple[dict[str, str], set[str], set[str]]:
    """Return (code->abbr, scored_codes, all_known_codes) from CinC 2020 CSVs."""
    code_to_abbr: dict[str, str] = {}
    scored_codes: set[str] = set()
    all_codes: set[str] = set()
    for fname, is_scored in (("dx_mapping_scored.csv", True), ("dx_mapping_unscored.csv", False)):
        p = QUALITY_DIR / fname
        if p.exists():
            mdf = pd.read_csv(p)
            mdf.columns = mdf.columns.str.strip()
            for code, abbr in zip(mdf["SNOMED CT Code"].astype(str), mdf["Abbreviation"].astype(str)):
                c = code.strip()
                code_to_abbr.setdefault(c, abbr.strip())
                all_codes.add(c)
                if is_scored:
                    scored_codes.add(c)
    return code_to_abbr, scored_codes, all_codes


def main() -> None:
    t0 = time.time()
    code_to_abbr, scored_codes, known_codes = load_snomed_maps()
    records = [l.strip() for l in (DATA_DIR / "RECORDS").read_text().splitlines() if l.strip()]
    print(f"Scanning all {len(records):,} Georgia 12-Lead ECG records (500 Hz) across 16 threads ...")

    # Bandpass 5–20 Hz at 500 Hz for fast R-peak HR estimation
    b_qrs, a_qrs = sp_signal.butter(3, [5.0 / 250.0, 20.0 / 250.0], btype="bandpass")

    def inspect_record(rel_path: str) -> dict:
        rec_id = Path(rel_path).name
        subfolder = Path(rel_path).parent.name
        hea_path = DATA_DIR / (rel_path + ".hea")
        mat_path = DATA_DIR / (rel_path + ".mat")

        base = {
            "record_id": rec_id,
            "record_path": rel_path,
            "subfolder": subfolder,
            "age": np.nan,
            "age_raw": "",
            "sex": "",
            "n_leads": -1,
            "fs": -1,
            "n_samples": -1,
            "duration_s": np.nan,
            "snomed_codes": "",
            "label_names": "",
            "n_dx_codes": 0,
            "n_scored_codes": 0,
            "est_hr_bpm": np.nan,
            "median_rms_mv": np.nan,
            "max_abs_mv": np.nan,
            "einthoven_rmse_mv": np.nan,
            "goldberger_rmse_mv": np.nan,
            "lf_wander_pct": np.nan,
            "hf_noise_pct": np.nan,
            "mains_60hz_pct": np.nan,
            # Error flags
            "err_file_missing_or_corrupt": 0,
            "err_hdr_nonstandard_duration": 0,
            "err_hdr_nonstandard_fs_or_leads": 0,
            "err_age_missing": 0,
            "err_age_invalid_or_sentinel": 0,
            "err_age_pediatric": 0,
            "err_sex_missing_or_unknown": 0,
            "err_dx_empty": 0,
            "err_dx_unmapped_code": 0,
            "err_dx_no_scored_cinc2020": 0,
            "err_dx_nsr_pathology_conflict": 0,
            "err_rhythm_hr_mismatch": 0,
            "err_sig_nan_or_inf": 0,
            "err_sig_flatline_lead": 0,
            "err_sig_extreme_amplitude": 0,
            "err_sig_large_dc_offset": 0,
            "err_sig_lead_physics_violation": 0,
            "err_sig_excessive_wander_or_hf": 0,
        }

        if not hea_path.exists() or not mat_path.exists():
            base["err_file_missing_or_corrupt"] = 1
            return base

        # ── Parse .hea header ──────────────────────────────────────────
        try:
            lines = hea_path.read_text(errors="replace").splitlines()
            first = lines[0].split()
            n_leads = int(first[1])
            fs = int(float(first[2]))
            n_samples = int(first[3])
            lead_names = tuple(
                l.split()[-1] for l in lines[1 : 1 + n_leads] if l.strip() and not l.startswith("#")
            )
        except Exception:
            base["err_file_missing_or_corrupt"] = 1
            return base

        base["n_leads"] = n_leads
        base["fs"] = fs
        base["n_samples"] = n_samples
        base["duration_s"] = round(n_samples / fs, 3) if fs > 0 else np.nan
        if n_samples != 5000:
            base["err_hdr_nonstandard_duration"] = 1
        if fs != 500 or n_leads != 12 or tuple(l.upper() for l in lead_names) != tuple(l.upper() for l in EXPECTED_LEADS):
            base["err_hdr_nonstandard_fs_or_leads"] = 1

        age_raw = ""
        sex_raw = ""
        dx_codes: list[str] = []
        for l in lines:
            s = l.strip()
            if not s.startswith("#"):
                continue
            m = re.match(r"#\s*(Age|Sex|Dx)\s*:\s*(.*)", s, re.IGNORECASE)
            if not m:
                continue
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "age":
                age_raw = val
            elif key == "sex":
                sex_raw = val
            elif key == "dx":
                dx_codes = [c.strip() for c in val.split(",") if c.strip()]

        base["age_raw"] = age_raw
        base["sex"] = sex_raw

        if not age_raw or age_raw.lower() in ("nan", "unknown", "none"):
            base["err_age_missing"] = 1
        else:
            try:
                age_val = float(age_raw)
                base["age"] = age_val
                if age_val <= 0 or age_val > 100:
                    base["err_age_invalid_or_sentinel"] = 1
                elif age_val < 18:
                    base["err_age_pediatric"] = 1
            except ValueError:
                base["err_age_missing"] = 1

        if sex_raw.lower() not in ("male", "female", "m", "f"):
            base["err_sex_missing_or_unknown"] = 1

        dx_set = set(dx_codes)
        base["snomed_codes"] = "|".join(dx_codes)
        base["label_names"] = "|".join(code_to_abbr.get(c, c) for c in dx_codes)
        base["n_dx_codes"] = len(dx_codes)
        base["n_scored_codes"] = sum(1 for c in dx_codes if c in scored_codes)

        if len(dx_codes) == 0:
            base["err_dx_empty"] = 1
        else:
            if any(c not in known_codes for c in dx_codes):
                base["err_dx_unmapped_code"] = 1
            if base["n_scored_codes"] == 0:
                base["err_dx_no_scored_cinc2020"] = 1
            if SNOMED_NSR in dx_set and bool(dx_set & SNOMED_NSR_CONFLICTS):
                base["err_dx_nsr_pathology_conflict"] = 1

        # ── Load & audit 500 Hz waveform ───────────────────────────────
        try:
            sig, _ = wfdb.rdsamp(str(DATA_DIR / rel_path))
        except Exception:
            base["err_file_missing_or_corrupt"] = 1
            return base

        has_nan_inf = bool(np.isnan(sig).any() or np.isinf(sig).any())
        if has_nan_inf:
            base["err_sig_nan_or_inf"] = 1
            sig = np.nan_to_num(sig)

        if sig.ndim != 2 or sig.shape[1] != 12 or sig.shape[0] < 100:
            base["err_file_missing_or_corrupt"] = 1
            return base

        lead_std = sig.std(axis=0)
        max_abs = float(np.abs(sig).max())
        max_dc = float(np.abs(sig.mean(axis=0)).max())
        base["median_rms_mv"] = round(float(np.median(np.sqrt(np.mean(sig ** 2, axis=0)))), 5)
        base["max_abs_mv"] = round(max_abs, 4)

        # Longest flatline run (>= 500 ms = fs // 2 samples)
        zero_diff = np.diff(sig, axis=0) == 0.0
        flat_thresh = max(int(fs * 0.5), 50)
        max_run = 0
        for j in range(12):
            z = zero_diff[:, j]
            if z.any():
                padded = np.concatenate([[False], z, [False]])
                starts = np.flatnonzero(padded[1:] & ~padded[:-1])
                ends = np.flatnonzero(~padded[1:] & padded[:-1])
                run = int((ends - starts).max())
                if run > max_run:
                    max_run = run

        if lead_std.min() < 0.005 or max_run >= flat_thresh:
            base["err_sig_flatline_lead"] = 1
        if max_abs > 10.0:
            base["err_sig_extreme_amplitude"] = 1
        if max_dc > 0.5:
            base["err_sig_large_dc_offset"] = 1

        # Biophysical limb-lead identities: Einthoven & Goldberger
        einth_rmse = float(np.sqrt(np.mean((sig[:, 1] - (sig[:, 0] + sig[:, 2])) ** 2)))
        gold_rmse = float(np.sqrt(np.mean((sig[:, 3] + sig[:, 4] + sig[:, 5]) ** 2)))
        base["einthoven_rmse_mv"] = round(einth_rmse, 6)
        base["goldberger_rmse_mv"] = round(gold_rmse, 6)
        if einth_rmse > 0.05 or gold_rmse > 0.05:
            base["err_sig_lead_physics_violation"] = 1

        # Spectral quality (baseline wander <0.5 Hz, high-freq >45 Hz, US 60 Hz mains hum)
        nperseg = min(1024, sig.shape[0])
        freqs, pxx = sp_signal.welch(sig, fs=float(fs), nperseg=nperseg, axis=0)
        tot = pxx.sum(axis=0) + 1e-12
        lf_pct = float(np.median(pxx[freqs < 0.5].sum(axis=0) / tot) * 100.0)
        hf_pct = float(np.median(pxx[freqs > 45.0].sum(axis=0) / tot) * 100.0)
        m60_pct = float(np.median(pxx[(freqs >= 58.0) & (freqs <= 62.0)].sum(axis=0) / tot) * 100.0)
        base["lf_wander_pct"] = round(lf_pct, 3)
        base["hf_noise_pct"] = round(hf_pct, 3)
        base["mains_60hz_pct"] = round(m60_pct, 3)
        if lf_pct > 35.0 or hf_pct > 25.0:
            base["err_sig_excessive_wander_or_hf"] = 1

        # Heart rate estimation on Lead II (500 Hz)
        if fs == 500 and sig.shape[0] >= 1000:
            s_ii = sig[:, 1] - np.median(sig[:, 1])
            e_ii = sp_signal.filtfilt(b_qrs, a_qrs, s_ii) ** 2
            pks, _ = sp_signal.find_peaks(
                e_ii, distance=125, height=max(float(np.percentile(e_ii, 85)) * 1.4, 1e-4)
            )
            if len(pks) >= 4:
                hr = float(60.0 / (np.median(np.diff(pks)) / 500.0))
                base["est_hr_bpm"] = round(hr, 2)
                if (bool(dx_set & SNOMED_SB) and hr > 68.0) or (bool(dx_set & SNOMED_STACH) and hr < 92.0):
                    base["err_rhythm_hr_mismatch"] = 1

        return base

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(inspect_record, records, chunksize=64))

    out = pd.DataFrame(rows).set_index("record_id")
    flag_cols = [c for c in out.columns if c.startswith("err_")]
    for c in flag_cols:
        out[c] = out[c].astype(np.int8)

    out["n_error_flags"] = out[flag_cols].sum(axis=1).astype(np.int8)

    severe_cols = [
        "err_file_missing_or_corrupt",
        "err_hdr_nonstandard_fs_or_leads",
        "err_dx_empty",
        "err_sig_nan_or_inf",
        "err_sig_flatline_lead",
        "err_sig_extreme_amplitude",
        "err_sig_lead_physics_violation",
    ]
    moderate_cols = [
        "err_hdr_nonstandard_duration",
        "err_age_invalid_or_sentinel",
        "err_dx_unmapped_code",
        "err_dx_nsr_pathology_conflict",
        "err_rhythm_hr_mismatch",
        "err_sig_excessive_wander_or_hf",
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
