"""
Generate per-record metadata summary and data-quality error labels for all
45,152 Chapman-Shaoxing ECG records (PhysioNet ecg-arrhythmia 1.0.0).

Audits all nested WFDBRecords/*/*/*.hea headers across 16 threads:
  1. File & Header Integrity (err_file_*, err_hdr_*)
  2. Demographics            (err_age_*, err_sex_*)
  3. SNOMED-CT Label Quality (err_dx_*)

Usage:
    .venv/bin/python chapman_shaoxing/quality/generate_error_labels.py

Output:
    chapman_shaoxing/quality/error_labels.csv
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

QUALITY_DIR = Path(__file__).resolve().parent
CHAPMAN_DIR = QUALITY_DIR.parent
DATA_DIR = CHAPMAN_DIR / "data"
OUT_CSV = QUALITY_DIR / "error_labels.csv"

EXPECTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")

SNOMED_SR = "426783006"
SNOMED_SR_CONFLICTS = {
    "164889003",  # AFIB
    "164890007",  # AF (Atrial Flutter)
    "426177001",  # SB
    "427084000",  # ST
    "426761007",  # SVT
}


def expand_records(root: Path) -> list[str]:
    out: list[str] = []
    top = [l.strip() for l in (root / "RECORDS").read_text().splitlines() if l.strip()]
    for entry in top:
        clean = entry.rstrip("/")
        nested = root / clean / "RECORDS"
        if nested.is_file():
            for rec in nested.read_text().splitlines():
                rec = rec.strip()
                if rec:
                    out.append(f"{clean}/{rec}")
        else:
            out.append(clean)
    return out


def main() -> None:
    t0 = time.time()
    records = expand_records(DATA_DIR)

    # Load ConditionNames_SNOMED-CT.csv
    cond_df = pd.read_csv(DATA_DIR / "ConditionNames_SNOMED-CT.csv")
    cond_df.columns = cond_df.columns.str.strip()
    code_to_abbr: dict[str, str] = {}
    for code, abbr in zip(cond_df["Snomed_CT"].astype(str), cond_df["Acronym Name"].astype(str)):
        code_to_abbr.setdefault(code.strip(), abbr.strip())
    known_codes = set(code_to_abbr.keys())

    print(f"Scanning all {len(records):,} Chapman-Shaoxing .hea headers across 16 threads ...")

    def inspect_hea(rel_path: str) -> dict:
        rec_id = Path(rel_path).name
        subfolder = Path(rel_path).parent.name
        hea_p = DATA_DIR / (rel_path + ".hea")
        mat_p = DATA_DIR / (rel_path + ".mat")

        row = {
            "record_id": rec_id,
            "record_path": rel_path,
            "subfolder": subfolder,
            "age": np.nan,
            "sex": "",
            "n_leads": -1,
            "fs": -1,
            "n_samples": -1,
            "duration_s": np.nan,
            "snomed_codes": "",
            "label_names": "",
            "n_dx_codes": 0,
            "err_file_missing_or_corrupt": 0,
            "err_hdr_nonstandard_duration": 0,
            "err_hdr_nonstandard_fs_or_leads": 0,
            "err_age_missing": 0,
            "err_age_invalid_or_sentinel": 0,
            "err_age_pediatric": 0,
            "err_sex_missing_or_unknown": 0,
            "err_dx_empty": 0,
            "err_dx_unmapped_code": 0,
            "err_dx_sr_arrhythmia_conflict": 0,
        }

        if not hea_p.exists() or not mat_p.exists() or mat_p.stat().st_size == 0:
            row["err_file_missing_or_corrupt"] = 1
            return row

        try:
            lines = hea_p.read_text(errors="replace").splitlines()
            first = lines[0].split()
            n_leads = int(first[1])
            fs = int(float(first[2]))
            n_samples = int(first[3])
            lead_names = tuple(
                l.split()[-1].upper() for l in lines[1 : 1 + n_leads] if l.strip() and not l.startswith("#")
            )
        except Exception:
            row["err_file_missing_or_corrupt"] = 1
            return row

        row["n_leads"] = n_leads
        row["fs"] = fs
        row["n_samples"] = n_samples
        row["duration_s"] = round(n_samples / fs, 3) if fs > 0 else np.nan
        if n_samples != 5000:
            row["err_hdr_nonstandard_duration"] = 1
        if fs != 500 or n_leads != 12 or lead_names != EXPECTED_LEADS:
            row["err_hdr_nonstandard_fs_or_leads"] = 1

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
            k, v = m.group(1).lower(), m.group(2).strip()
            if k == "age":
                age_raw = v
            elif k == "sex":
                sex_raw = v
            elif k == "dx":
                dx_codes = [c.strip() for c in v.split(",") if c.strip()]

        row["sex"] = sex_raw
        if not age_raw or age_raw.lower() in ("nan", "unknown", "none"):
            row["err_age_missing"] = 1
        else:
            try:
                aval = float(age_raw)
                row["age"] = aval
                if aval <= 0 or aval > 100:
                    row["err_age_invalid_or_sentinel"] = 1
                elif aval < 18:
                    row["err_age_pediatric"] = 1
            except ValueError:
                row["err_age_missing"] = 1

        if sex_raw.lower() not in ("male", "female", "m", "f"):
            row["err_sex_missing_or_unknown"] = 1

        dx_set = set(dx_codes)
        row["snomed_codes"] = "|".join(dx_codes)
        row["label_names"] = "|".join(code_to_abbr.get(c, c) for c in dx_codes)
        row["n_dx_codes"] = len(dx_codes)

        if len(dx_codes) == 0:
            row["err_dx_empty"] = 1
        else:
            if any(c not in known_codes for c in dx_codes):
                row["err_dx_unmapped_code"] = 1
            if SNOMED_SR in dx_set and bool(dx_set & SNOMED_SR_CONFLICTS):
                row["err_dx_sr_arrhythmia_conflict"] = 1

        return row

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(inspect_hea, records, chunksize=128))

    out = pd.DataFrame(rows).set_index("record_id")
    flag_cols = [c for c in out.columns if c.startswith("err_")]
    for c in flag_cols:
        out[c] = out[c].astype(np.int8)
    out["n_error_flags"] = out[flag_cols].sum(axis=1).astype(np.int8)

    severe_cols = ["err_file_missing_or_corrupt", "err_hdr_nonstandard_fs_or_leads", "err_dx_empty"]
    moderate_cols = [
        "err_hdr_nonstandard_duration",
        "err_age_invalid_or_sentinel",
        "err_dx_unmapped_code",
        "err_dx_sr_arrhythmia_conflict",
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
    print(f"\nSaved {len(out):,} rows × {len(out.columns)} cols to {OUT_CSV} in {elapsed:.1f}s ({OUT_CSV.stat().st_size / (1024*1024):.2f} MB)")
    print("\n=== Severity Tier Breakdown ===")
    print(out["severity_tier"].value_counts().sort_index())
    print("\n=== Individual Error Flag Counts ===")
    for c in flag_cols:
        cnt = int(out[c].sum())
        print(f"  {c:<34}: {cnt:6,d} ({cnt / len(out) * 100:5.2f}%)")


if __name__ == "__main__":
    main()
