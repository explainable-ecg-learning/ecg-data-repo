"""PyTorch Dataset loader for the combined, cleaned, multi-source 12-lead ECG corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CombinedECGDataset(Dataset):
    """PyTorch Dataset reading sharded HDF5 files produced by ``combined/build_combined.py``.

    Supports the repository-wide quality-filtering API (``load_error_labels``,
    ``max_error_severity``, ``exclude_error_flags``) backed by
    ``combined/quality/error_labels.csv``.

    Each sample returned by ``__getitem__`` is a dict with:

    * ``signal``           : float32 ndarray ``(12, T)``
    * ``source_dataset``   : str (``ptbxl``, ``mimic_iv_ecg``, ``georgia_ecg``, ``chapman_shaoxing``, ``code15``)
    * ``source_record_id`` : str
    * ``patient_id``       : str (prefixed by source dataset for global uniqueness)
    * ``labels``           : str (source diagnostic labels / codes)
    * ``meta``             : dict with ``global_id``, ``age``, ``sex``, ``original_fs``, ``severity_tier``, ``split``
    * ``error_labels``     : dict (when ``load_error_labels=True``)
    """

    LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    PACKAGE_DIR = Path(__file__).resolve().parent
    QUALITY_DIR = PACKAGE_DIR / "quality"
    DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
    ERROR_LABELS_CSV = QUALITY_DIR / "error_labels.csv.gz"

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_DIR,
        split: Optional[str] = None,
        sources: Optional[list[str]] = None,
        load_error_labels: bool = False,
        max_error_severity: Optional[int] = None,
        exclude_error_flags: Optional[list[str]] = None,
        error_labels_path: str | Path = ERROR_LABELS_CSV,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if max_error_severity is not None and max_error_severity not in (0, 1, 2, 3):
            raise ValueError("max_error_severity must be 0, 1, 2, 3, or None")

        self.root = Path(root)
        self.load_error_labels = load_error_labels
        self.transform = transform
        self._handles: dict[str, h5py.File] = {}
        self._error_df: Optional[pd.DataFrame] = None
        self._error_flag_cols: list[str] = []

        need_err = load_error_labels or (max_error_severity is not None) or bool(exclude_error_flags)
        if need_err:
            err_path = Path(error_labels_path)
            if not err_path.exists():
                raise FileNotFoundError(
                    f"Error labels file not found at {err_path}. "
                    "Run `python combined/quality/generate_error_labels.py` to generate it."
                )
            self._error_df = pd.read_csv(err_path, index_col="global_id")
            self._error_df["error_tags"] = self._error_df["error_tags"].fillna("")
            self._error_flag_cols = [c for c in self._error_df.columns if c.startswith("err_")]

        meta_path = self.root / "metadata.csv.gz"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Combined metadata not found at {meta_path}. "
                "Run `python combined/build_combined.py` first."
            )

        df = pd.read_csv(meta_path)
        if split is not None:
            if split not in ("train", "val", "test"):
                raise ValueError("split must be 'train', 'val', or 'test', or None")
            df = df[df["split"] == split]
        if sources is not None:
            df = df[df["source_dataset"].isin(sources)]

        if self._error_df is not None and (max_error_severity is not None or exclude_error_flags):
            err_sub = self._error_df.reindex(df["global_id"])
            keep = np.ones(len(df), dtype=bool)
            if max_error_severity is not None:
                tier_num = err_sub["severity_tier"].str.slice(0, 1).astype(int).to_numpy()
                keep &= tier_num <= max_error_severity
            if exclude_error_flags:
                for col in exclude_error_flags:
                    if col not in err_sub.columns:
                        raise ValueError(f"Unknown error flag '{col}'")
                    keep &= err_sub[col].to_numpy() == 0
            df = df[keep]

        self._df = df.reset_index(drop=True)

        cfg_path = self.root / "build_config.json"
        self.build_config: dict = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    def _get_shard(self, shard_file: str) -> h5py.File:
        if shard_file not in self._handles:
            self._handles[shard_file] = h5py.File(self.root / shard_file, "r")
        return self._handles[shard_file]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.iloc[idx]
        gid = int(row["global_id"])
        f = self._get_shard(str(row["shard_file"]))
        sig = np.asarray(f["signals"][int(row["shard_idx"])], dtype=np.float32)

        if self.transform is not None:
            sig = self.transform(sig)

        sample = {
            "signal": sig,
            "source_dataset": str(row["source_dataset"]),
            "source_record_id": str(row["source_record_id"]),
            "patient_id": str(row["patient_id"]),
            "labels": "" if pd.isna(row["raw_labels"]) else str(row["raw_labels"]),
            "meta": {
                "global_id": gid,
                "age": None if pd.isna(row["age"]) else float(row["age"]),
                "sex": "" if pd.isna(row["sex"]) else str(row["sex"]),
                "original_fs": int(row["original_fs"]),
                "severity_tier": str(row["severity_tier"]),
                "split": str(row["split"]),
            },
        }
        if self.load_error_labels and self._error_df is not None:
            erow = self._error_df.loc[gid]
            tags_str = str(erow["error_tags"])
            sample["error_labels"] = {
                "severity_tier": str(erow["severity_tier"]),
                "recommended_action": str(erow["recommended_action"]),
                "n_error_flags": int(erow["n_error_flags"]),
                "error_tags": tags_str.split("|") if tags_str else [],
                "flags": {c: int(erow[c]) for c in self._error_flag_cols},
            }
        return sample

    def __del__(self) -> None:
        for f in self._handles.values():
            try:
                f.close()
            except Exception:
                pass

    @classmethod
    def train(cls, root: str | Path = DEFAULT_DATA_DIR, **kwargs) -> "CombinedECGDataset":
        return cls(root=root, split="train", **kwargs)

    @classmethod
    def val(cls, root: str | Path = DEFAULT_DATA_DIR, **kwargs) -> "CombinedECGDataset":
        return cls(root=root, split="val", **kwargs)

    @classmethod
    def test(cls, root: str | Path = DEFAULT_DATA_DIR, **kwargs) -> "CombinedECGDataset":
        return cls(root=root, split="test", **kwargs)
