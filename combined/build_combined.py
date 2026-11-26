"""Build one unified, cleaned, resampled, denoised, and smoothed 12-lead ECG dataset.

Combines clean records (default: ``0_clean`` only) from all five benchmark datasets
(PTB-XL, MIMIC-IV-ECG, Georgia 12-Lead ECG, Chapman-Shaoxing, and CODE-15%) into
sharded HDF5 files + a patient-stratified ``metadata.csv.gz`` under ``combined/data/``.

Uses multi-process workers (``--workers``) for high-throughput WFDB/HDF5 reading
and SciPy signal preprocessing.

Example usage:
    # Full build (0_clean only, 500 Hz, 10 s, float16 + lzf HDF5 shards)
    .venv/bin/python combined/build_combined.py --workers 16

    # Quick smoke test with 50 records per dataset
    .venv/bin/python combined/build_combined.py --max-per-dataset 50 --output-dir combined/data_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from combined.preprocessing import PreprocessConfig, preprocess_ecg

ALL_DATASETS = ("ptbxl", "mimic_iv_ecg", "georgia_ecg", "chapman_shaoxing", "code15")

# Global worker config (set via ProcessPoolExecutor initializer)
_WORKER_CFG: PreprocessConfig = PreprocessConfig()
_WORKER_DTYPE: np.dtype = np.float16


def _init_worker(cfg: PreprocessConfig, dtype_str: str) -> None:
    global _WORKER_CFG, _WORKER_DTYPE
    _WORKER_CFG = cfg
    _WORKER_DTYPE = np.float16 if dtype_str == "float16" else np.float32


def _process_wfdb_task(task: tuple) -> tuple[np.ndarray | None, dict]:
    """Worker: read one WFDB record, preprocess, and cast to target storage dtype."""
    rec_path, default_fs, meta = task
    if not Path(rec_path + ".hea").exists():
        return None, meta
    try:
        sig, fields = wfdb.rdsamp(rec_path)
        orig_fs = int(fields.get("fs", default_fs)) if fields else default_fs
    except Exception:
        return None, meta
    proc = preprocess_ecg(sig.T, orig_fs=orig_fs, cfg=_WORKER_CFG, trim_zero_pad=False)
    if proc is None:
        return None, meta
    meta["original_fs"] = orig_fs
    return proc.astype(_WORKER_DTYPE, copy=False), meta


def _process_code15_part_chunk(task: tuple) -> list[tuple[np.ndarray, dict]]:
    """Worker: read and preprocess a chunk of rows from a single CODE-15% HDF5 part."""
    part_path, items = task  # items: list of (local_idx, meta)
    out: list[tuple[np.ndarray, dict]] = []
    with h5py.File(part_path, "r") as f:
        tracings = f["tracings"]
        for local_idx, meta in items:
            sig = np.asarray(tracings[local_idx]).T  # (12, 4096)
            proc = preprocess_ecg(sig, orig_fs=400, cfg=_WORKER_CFG, trim_zero_pad=True)
            if proc is not None:
                out.append((proc.astype(_WORKER_DTYPE, copy=False), meta))
    return out


def _tier_allowed(series: pd.Series, max_severity: int) -> pd.Series:
    return series.astype(str).str.slice(0, 1).astype(int) <= max_severity


def _build_wfdb_tasks(ds_key: str, max_severity: int, max_records: int | None) -> list[tuple]:
    tasks: list[tuple] = []
    if ds_key == "ptbxl":
        err = pd.read_csv(REPO_ROOT / "ptbxl" / "quality" / "error_labels.csv", usecols=["ecg_id", "severity_tier"])
        err = err[_tier_allowed(err["severity_tier"], max_severity)]
        keep_map = dict(zip(err["ecg_id"], err["severity_tier"]))
        db = pd.read_csv(
            REPO_ROOT / "ptbxl" / "data" / "ptbxl_database.csv",
            usecols=["ecg_id", "patient_id", "age", "sex", "scp_codes", "filename_hr"],
        )
        db = db[db["ecg_id"].isin(keep_map)]
        if max_records is not None:
            db = db.head(max_records)
        base = str(REPO_ROOT / "ptbxl" / "data")
        for row in db.itertuples(index=False):
            age = None if pd.isna(row.age) or float(row.age) >= 200 else float(row.age)
            sex = "M" if row.sex == 0 else ("F" if row.sex == 1 else "")
            tasks.append((
                f"{base}/{row.filename_hr}",
                500,
                {
                    "source_dataset": "ptbxl",
                    "source_record_id": str(int(row.ecg_id)),
                    "patient_id": f"ptbxl:{int(row.patient_id)}",
                    "age": age,
                    "sex": sex,
                    "original_fs": 500,
                    "severity_tier": keep_map[row.ecg_id],
                    "raw_labels": str(row.scp_codes),
                },
            ))
    elif ds_key == "mimic_iv_ecg":
        err = pd.read_csv(
            REPO_ROOT / "mimic_iv_ecg" / "quality" / "error_labels.csv.gz",
            usecols=["study_id", "subject_id", "severity_tier"],
        )
        err = err[_tier_allowed(err["severity_tier"], max_severity)]
        keep_map = dict(zip(err["study_id"], err["severity_tier"]))
        rec = pd.read_csv(
            REPO_ROOT / "mimic_iv_ecg" / "data" / "record_list.csv",
            usecols=["study_id", "subject_id", "path"],
        )
        rec = rec[rec["study_id"].isin(keep_map)]
        if max_records is not None:
            rec = rec.head(max_records)
        base = str(REPO_ROOT / "mimic_iv_ecg" / "data")
        for row in rec.itertuples(index=False):
            tasks.append((
                f"{base}/{row.path}",
                500,
                {
                    "source_dataset": "mimic_iv_ecg",
                    "source_record_id": str(int(row.study_id)),
                    "patient_id": f"mimic_iv_ecg:{int(row.subject_id)}",
                    "age": None,
                    "sex": "",
                    "original_fs": 500,
                    "severity_tier": keep_map[row.study_id],
                    "raw_labels": "",
                },
            ))
    elif ds_key in ("georgia_ecg", "chapman_shaoxing"):
        err = pd.read_csv(
            REPO_ROOT / ds_key / "quality" / "error_labels.csv",
            usecols=["record_id", "record_path", "age", "sex", "fs", "label_names", "severity_tier"],
        )
        err = err[_tier_allowed(err["severity_tier"], max_severity)]
        if max_records is not None:
            err = err.head(max_records)
        base = str(REPO_ROOT / ds_key / "data")
        for row in err.itertuples(index=False):
            age = None if pd.isna(row.age) or float(row.age) <= 0 or float(row.age) > 120 else float(row.age)
            s_raw = "" if pd.isna(row.sex) else str(row.sex).strip().lower()
            sex = "M" if s_raw.startswith("m") else ("F" if s_raw.startswith("f") else "")
            fs_val = 500 if pd.isna(row.fs) else int(row.fs)
            tasks.append((
                f"{base}/{row.record_path}",
                fs_val,
                {
                    "source_dataset": ds_key,
                    "source_record_id": str(row.record_id),
                    "patient_id": f"{ds_key}:{row.record_id}",
                    "age": age,
                    "sex": sex,
                    "original_fs": fs_val,
                    "severity_tier": str(row.severity_tier),
                    "raw_labels": "" if pd.isna(row.label_names) else str(row.label_names),
                },
            ))
    return tasks


def _build_code15_chunks(max_severity: int, max_records: int | None, chunk_size: int = 500) -> list[tuple]:
    err = pd.read_csv(
        REPO_ROOT / "code15" / "quality" / "error_labels.csv.gz",
        usecols=["exam_id", "severity_tier"],
    )
    err = err[_tier_allowed(err["severity_tier"], max_severity)]
    keep_map = dict(zip(err["exam_id"], err["severity_tier"]))

    label_cols = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
    exams = pd.read_csv(
        REPO_ROOT / "code15" / "data" / "exams.csv",
        usecols=["exam_id", "patient_id", "age", "is_male", "normal_ecg"] + label_cols,
    )
    exams = exams[exams["exam_id"].isin(keep_map)].set_index("exam_id")

    chunks: list[tuple] = []
    total = 0
    for part_path in sorted((REPO_ROOT / "code15" / "data").glob("exams_part*.hdf5")):
        with h5py.File(part_path, "r") as f:
            ids = np.array(f["exam_id"])
        part_items = []
        for local_idx, eid in enumerate(ids):
            eid_int = int(eid)
            if eid_int == 0 or eid_int not in exams.index:
                continue
            row = exams.loc[eid_int]
            active = [c for c in label_cols if int(row[c]) == 1]
            if not active and bool(row["normal_ecg"]):
                active = ["NORMAL"]
            part_items.append((
                int(local_idx),
                {
                    "source_dataset": "code15",
                    "source_record_id": str(eid_int),
                    "patient_id": f"code15:{int(row['patient_id'])}",
                    "age": None if pd.isna(row["age"]) else float(row["age"]),
                    "sex": "M" if bool(row["is_male"]) else "F",
                    "original_fs": 400,
                    "severity_tier": keep_map[eid_int],
                    "raw_labels": "|".join(active),
                },
            ))
            total += 1
            if max_records is not None and total >= max_records:
                break
        for i in range(0, len(part_items), chunk_size):
            chunks.append((str(part_path), part_items[i : i + chunk_size]))
        if max_records is not None and total >= max_records:
            break
    return chunks


def _assign_patient_splits(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42) -> pd.Series:
    """Stratified-by-source, patient-disjoint train / val / test split."""
    rng = np.random.default_rng(seed)
    patient_to_split: dict[str, str] = {}
    for _, grp in df.groupby("source_dataset", sort=True):
        pats = np.array(sorted(grp["patient_id"].unique()))
        rng.shuffle(pats)
        n = len(pats)
        n_test = max(1, int(round(n * test_frac))) if n >= 5 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
        for p in pats[:n_test]:
            patient_to_split[p] = "test"
        for p in pats[n_test : n_test + n_val]:
            patient_to_split[p] = "val"
        for p in pats[n_test + n_val :]:
            patient_to_split[p] = "train"
    return df["patient_id"].map(patient_to_split)


def _flush_shard(
    out_dir: Path,
    shard_idx: int,
    buf_signals: list[np.ndarray],
    buf_ids: list[int],
    dtype: np.dtype,
    compression: str | None,
) -> str:
    shard_name = f"combined_part{shard_idx:03d}.hdf5"
    arr = np.stack(buf_signals, axis=0).astype(dtype, copy=False)
    ids = np.array(buf_ids, dtype=np.int64)
    with h5py.File(out_dir / shard_name, "w") as f:
        f.create_dataset("signals", data=arr, compression=compression, chunks=(min(64, len(arr)), 12, arr.shape[2]))
        f.create_dataset("global_id", data=ids)
    return shard_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a unified, cleaned, preprocessed 12-lead ECG dataset.")
    p.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS), choices=list(ALL_DATASETS),
                   help="Which source datasets to include (default: all 5).")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "combined" / "data",
                   help="Output directory for HDF5 shards and metadata.csv.gz.")
    p.add_argument("--max-severity", type=int, default=0, choices=[0, 1, 2],
                   help="Maximum allowed error severity tier (default: 0 = 0_clean only).")
    p.add_argument("--target-fs", type=int, default=500, help="Target sampling rate in Hz (default: 500).")
    p.add_argument("--target-duration", type=float, default=10.0, help="Target duration in seconds (default: 10.0).")
    p.add_argument("--bandpass-low", type=float, default=0.5, help="High-pass cutoff in Hz (default: 0.5).")
    p.add_argument("--bandpass-high", type=float, default=40.0, help="Low-pass cutoff in Hz (default: 40.0).")
    p.add_argument("--no-smooth", action="store_true", help="Disable Savitzky-Golay smoothing.")
    p.add_argument("--normalize", choices=["none", "zscore"], default="none",
                   help="Per-lead normalization (default: none = calibrated mV).")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                   help="Storage dtype inside HDF5 shards (default: float16).")
    p.add_argument("--compression", choices=["none", "lzf", "gzip"], default="lzf",
                   help="HDF5 compression filter (default: lzf).")
    p.add_argument("--shard-size", type=int, default=25000, help="Records per HDF5 shard (default: 25000).")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8),
                   help="Number of parallel worker processes (default: min(16, cpu_count)).")
    p.add_argument("--max-per-dataset", type=int, default=None,
                   help="Optional cap per dataset (useful for fast smoke tests / dev builds).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for patient-stratified splits.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    cfg = PreprocessConfig(
        target_fs=args.target_fs,
        target_duration_s=args.target_duration,
        bandpass_low_hz=args.bandpass_low,
        bandpass_high_hz=args.bandpass_high,
        smooth=not args.no_smooth,
        normalize=args.normalize,
    )
    dtype = np.float16 if args.dtype == "float16" else np.float32
    compression = None if args.compression == "none" else args.compression

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale shards from any prior build in out_dir
    for old in out_dir.glob("combined_part*.hdf5"):
        old.unlink()

    meta_rows: list[dict] = []
    buf_signals: list[np.ndarray] = []
    buf_ids: list[int] = []
    shard_num = 0
    global_id = 0
    stats: dict[str, dict[str, int]] = {}

    def _append_record(sig_arr: np.ndarray, meta: dict) -> None:
        nonlocal shard_num, global_id
        shard_name = f"combined_part{shard_num:03d}.hdf5"
        local_idx = len(buf_signals)
        buf_signals.append(sig_arr)
        buf_ids.append(global_id)
        meta["global_id"] = global_id
        meta["shard_file"] = shard_name
        meta["shard_idx"] = local_idx
        meta_rows.append(meta)
        global_id += 1

        if len(buf_signals) >= args.shard_size:
            _flush_shard(out_dir, shard_num, buf_signals, buf_ids, dtype, compression)
            print(f"  -> Wrote {shard_name} ({len(buf_signals):,} records, total={global_id:,})", flush=True)
            buf_signals.clear()
            buf_ids.clear()
            shard_num += 1

    print(f"Starting combined build with {args.workers} workers (target={args.target_fs} Hz, {args.target_duration}s, {args.dtype}/{args.compression}) ...", flush=True)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(cfg, args.dtype),
    ) as pool:
        for ds_key in args.datasets:
            ds_t0 = time.time()
            kept, rejected = 0, 0
            if ds_key == "code15":
                chunks = _build_code15_chunks(args.max_severity, args.max_per_dataset)
                total_cand = sum(len(c[1]) for c in chunks)
                print(f"[code15] Processing {total_cand:,} clean candidates across {len(chunks)} chunks ...", flush=True)
                for chunk_res in pool.map(_process_code15_part_chunk, chunks):
                    for sig_arr, meta in chunk_res:
                        _append_record(sig_arr, meta)
                        kept += 1
                rejected = total_cand - kept
            else:
                tasks = _build_wfdb_tasks(ds_key, args.max_severity, args.max_per_dataset)
                print(f"[{ds_key}] Processing {len(tasks):,} clean candidates across {args.workers} workers ...", flush=True)
                for idx_t, (sig_arr, meta) in enumerate(pool.map(_process_wfdb_task, tasks, chunksize=64), 1):
                    if sig_arr is None:
                        rejected += 1
                    else:
                        _append_record(sig_arr, meta)
                        kept += 1
                    if idx_t % 50000 == 0:
                        print(f"  [{ds_key}] progress: {idx_t:,}/{len(tasks):,} (kept={kept:,}, skipped={rejected:,}, {time.time() - ds_t0:.0f}s)", flush=True)

            stats[ds_key] = {"kept": kept, "skipped_missing_or_qc": rejected}
            print(f"[{ds_key}] Done in {time.time() - ds_t0:.1f}s: kept={kept:,}, skipped={rejected:,}", flush=True)

    if buf_signals:
        shard_name = _flush_shard(out_dir, shard_num, buf_signals, buf_ids, dtype, compression)
        print(f"  -> Wrote {shard_name} ({len(buf_signals):,} records, total={global_id:,})", flush=True)
        shard_num += 1

    meta_df = pd.DataFrame(meta_rows)
    meta_df["split"] = _assign_patient_splits(meta_df, seed=args.seed)
    meta_path = out_dir / "metadata.csv.gz"
    meta_df.to_csv(meta_path, index=False, compression="gzip")

    build_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": int(len(meta_df)),
        "total_patients": int(meta_df["patient_id"].nunique()),
        "n_shards": int(shard_num),
        "storage_dtype": args.dtype,
        "compression": args.compression,
        "max_severity": args.max_severity,
        "max_per_dataset": args.max_per_dataset,
        "workers": args.workers,
        "preprocess_config": cfg.to_dict(),
        "per_dataset_stats": stats,
        "split_counts": meta_df["split"].value_counts().to_dict(),
    }
    (out_dir / "build_config.json").write_text(json.dumps(build_info, indent=2))

    print(
        f"\nCombined dataset build complete in {time.time() - t0:.1f}s!\n"
        f"  Output dir : {out_dir}\n"
        f"  Total ECGs : {len(meta_df):,} ({meta_df['patient_id'].nunique():,} unique patients, {shard_num} shards)\n"
        f"  Splits     : {build_info['split_counts']}"
    )


if __name__ == "__main__":
    main()
