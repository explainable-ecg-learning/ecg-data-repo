"""Generate post-build per-record quality & error labels for the combined ECG dataset.

Audits every record in ``combined/data/`` (``metadata.csv.gz`` + ``combined_part*.hdf5``)
in parallel across HDF5 shards and writes ``combined/quality/error_labels.csv.gz``.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED_DIR = REPO_ROOT / "combined"
DATA_DIR = COMBINED_DIR / "data"
QUALITY_DIR = COMBINED_DIR / "quality"
OUT_CSV = QUALITY_DIR / "error_labels.csv.gz"


def _audit_shard(task: tuple) -> list[dict]:
    shard_path, rows_list, fs, clip_mv = task
    out: list[dict] = []
    with h5py.File(shard_path, "r") as f:
        signals = f["signals"]
        for row in rows_list:
            sig = np.asarray(signals[int(row["shard_idx"])], dtype=np.float32)

            has_nan = int(not np.isfinite(sig).all())
            safe = np.nan_to_num(sig)
            lead_stds = np.std(safe, axis=1)
            flatline = int(np.any(lead_stds < 1e-4))
            max_abs = float(np.max(np.abs(safe)))
            clipped = int(max_abs >= 0.99 * clip_mv)
            rms_ii = float(np.sqrt(np.mean(safe[1] ** 2)))
            einth = float(np.sqrt(np.mean((safe[1] - (safe[0] + safe[2])) ** 2)))
            einth_bad = int(einth > 0.15)

            lead_ii = safe[1] - np.mean(safe[1])
            freqs, pxx = welch(lead_ii, fs=fs, nperseg=min(len(lead_ii), fs * 4))
            total = float(np.trapezoid(pxx, freqs)) + 1e-12
            lf_mask = freqs < 0.5
            m50_mask = (freqs >= 49.0) & (freqs <= 51.0)
            m60_mask = (freqs >= 59.0) & (freqs <= 61.0)
            hf_mask = freqs > 40.0
            lf_pct = float(np.trapezoid(pxx[lf_mask], freqs[lf_mask])) / total * 100.0
            m50_pct = float(np.trapezoid(pxx[m50_mask], freqs[m50_mask])) / total * 100.0
            m60_pct = float(np.trapezoid(pxx[m60_mask], freqs[m60_mask])) / total * 100.0
            mains_pct = max(m50_pct, m60_pct)
            hf_pct = float(np.trapezoid(pxx[hf_mask], freqs[hf_mask])) / total * 100.0

            res_wander = int(lf_pct > 5.0)
            res_mains = int(mains_pct > 2.0)
            res_hf = int(hf_pct > 5.0)

            age_missing = int(pd.isna(row["age"]))
            sex_missing = int(pd.isna(row["sex"]) or str(row["sex"]).strip() == "")
            lbl_missing = int(pd.isna(row["raw_labels"]) or str(row["raw_labels"]).strip() == "")

            flags = {
                "err_sig_nan_or_inf": has_nan,
                "err_sig_flatline_lead": flatline,
                "err_sig_clipped_amplitude": clipped,
                "err_sig_lead_physics_violation": einth_bad,
                "err_spec_residual_wander": res_wander,
                "err_spec_residual_mains": res_mains,
                "err_spec_residual_hf": res_hf,
                "err_meta_age_missing": age_missing,
                "err_meta_sex_missing": sex_missing,
                "err_meta_labels_missing": lbl_missing,
            }

            if has_nan or flatline:
                tier = "3_severe_error"
                action = "exclude_from_training"
            elif clipped or einth_bad or res_mains:
                tier = "2_moderate_anomaly"
                action = "review_or_downweight"
            elif res_wander or res_hf or age_missing or sex_missing or lbl_missing:
                tier = "1_minor_warning"
                action = "keep_with_caution"
            else:
                tier = "0_clean"
                action = "keep"

            active_tags = [k for k, v in flags.items() if v == 1]
            out.append({
                "global_id": int(row["global_id"]),
                "source_dataset": str(row["source_dataset"]),
                "split": str(row["split"]),
                "rms_ii_mv": round(rms_ii, 4),
                "max_abs_mv": round(max_abs, 4),
                "einthoven_rmse_mv": round(einth, 4),
                "lf_wander_pct": round(lf_pct, 3),
                "mains_50_60_pct": round(mains_pct, 3),
                "hf_gt40_pct": round(hf_pct, 3),
                **flags,
                "n_error_flags": sum(flags.values()),
                "severity_tier": tier,
                "recommended_action": action,
                "error_tags": "|".join(active_tags),
            })
    return out


def main() -> None:
    t0 = time.time()
    meta_path = DATA_DIR / "metadata.csv.gz"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Combined dataset metadata not found at {meta_path}. "
            "Run `python combined/build_combined.py` first."
        )

    meta = pd.read_csv(meta_path)
    cfg_path = DATA_DIR / "build_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    fs = int(cfg.get("preprocess_config", {}).get("target_fs", 500))
    clip_mv = float(cfg.get("preprocess_config", {}).get("clip_mv", 20.0))

    tasks = []
    for sf, grp in meta.groupby("shard_file", sort=True):
        tasks.append((str(DATA_DIR / str(sf)), grp.to_dict("records"), fs, clip_mv))

    workers = min(len(tasks), min(16, os.cpu_count() or 8))
    print(f"Auditing {len(meta):,} combined ECG records across {len(tasks)} shards ({workers} workers) ...", flush=True)

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, res in enumerate(pool.map(_audit_shard, tasks), 1):
            all_rows.extend(res)
            print(f"  -> Audited shard {i}/{len(tasks)} ({len(all_rows):,} records, {time.time() - t0:.1f}s)", flush=True)

    out_df = pd.DataFrame(all_rows).sort_values("global_id").reset_index(drop=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, compression="gzip")

    old_plain = QUALITY_DIR / "error_labels.csv"
    if old_plain.exists():
        old_plain.unlink()

    size_mb = OUT_CSV.stat().st_size / (1024 * 1024)
    print(f"Saved {len(out_df):,} rows × {out_df.shape[1]} cols to {OUT_CSV} in {time.time() - t0:.1f}s ({size_mb:.2f} MB)")
    print("\n=== Severity Tier Breakdown ===")
    print(out_df["severity_tier"].value_counts().sort_index())
    print("\n=== Individual Error Flag Counts ===")
    for c in [c for c in out_df.columns if c.startswith("err_")]:
        cnt = int(out_df[c].sum())
        print(f"  {c:34s}: {cnt:7,d} ({100 * cnt / len(out_df):5.2f}%)")


if __name__ == "__main__":
    main()
