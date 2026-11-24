from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CODE15Dataset(Dataset):
    """PyTorch Dataset for the CODE-15% ECG dataset (Zenodo).

    345,779 exams from 233,770 patients · 400 Hz · 4,096 samples (zero-padded
    to uniform length from 7 s or 10 s recordings) · 6 binary condition labels.

    Labels (all binary):
        ``1dAVb``, ``RBBB``, ``LBBB``, ``SB``, ``AF``, ``ST``

    Data is stored in 18 HDF5 part files (``exams_part0.hdf5`` …
    ``exams_part17.hdf5``) located inside the root directory.  Each file
    contains two datasets:

    * ``tracings`` : float32 tensor ``(N, 4096, 12)``
    * ``exam_id``  : int tensor ``(N,)``

    Metadata (age, sex, labels, …) lives in ``exams.csv``.

    Each sample returned by ``__getitem__`` is a dict with:

    * ``signal``  : float32 ndarray ``(12, 4096)``
    * ``labels``  : dict mapping each condition name to 0 / 1
    * ``meta``    : remaining ``exams.csv`` columns as a dict

    Args:
        root: Path to the CODE-15% root directory (contains ``exams.csv`` and
            the ``exams_part*.hdf5`` files).
        patient_ids: If given, only exams from these patient IDs are included.
        transform: Optional callable applied to the signal array.
    """

    LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    LABEL_COLS = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
    SAMPLING_RATE = 400
    SIGNAL_LENGTH = 4096
    PACKAGE_DIR = Path(__file__).resolve().parent
    QUALITY_DIR = PACKAGE_DIR / "quality"
    DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
    ERROR_LABELS_CSV = QUALITY_DIR / "error_labels.csv.gz"

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_DIR,
        patient_ids: Optional[list[int]] = None,
        only_available_parts: bool = True,
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
        self._error_df: Optional[pd.DataFrame] = None
        self._error_flag_cols: list[str] = []

        need_err = load_error_labels or (max_error_severity is not None) or bool(exclude_error_flags)
        if need_err:
            err_path = Path(error_labels_path)
            if not err_path.exists():
                raise FileNotFoundError(
                    f"Error labels file not found at {err_path}. "
                    "Run `python code15/quality/generate_error_labels.py` to generate it."
                )
            self._error_df = pd.read_csv(err_path, index_col="exam_id")
            self._error_df["error_tags"] = self._error_df["error_tags"].fillna("")
            self._error_flag_cols = [c for c in self._error_df.columns if c.startswith("err_")]

        df = pd.read_csv(self.root / "exams.csv")
        df.columns = df.columns.str.strip()

        if patient_ids is not None:
            df = df[df["patient_id"].isin(patient_ids)]

        # Build a lookup: exam_id → (part_file_path, index_within_file)
        self._exam_index: dict[int, tuple[Path, int]] = {}
        self._hdf5_handles: dict[Path, h5py.File] = {}
        self._build_index()

        if only_available_parts:
            df = df[df["exam_id"].isin(self._exam_index)]

        if self._error_df is not None and (max_error_severity is not None or exclude_error_flags):
            err_sub = self._error_df.reindex(df["exam_id"])
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

    def _build_index(self) -> None:
        """Map every exam_id to its HDF5 file and position within it."""
        part_files = sorted(self.root.glob("exams_part*.hdf5"))
        for part_path in part_files:
            with h5py.File(part_path, "r") as f:
                ids = np.array(f["exam_id"])
            for local_idx, eid in enumerate(ids):
                self._exam_index[int(eid)] = (part_path, local_idx)

    def _get_hdf5(self, path: Path) -> h5py.File:
        """Return a cached open HDF5 file handle (lazy open)."""
        if path not in self._hdf5_handles:
            self._hdf5_handles[path] = h5py.File(path, "r")
        return self._hdf5_handles[path]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.iloc[idx]
        exam_id = int(row["exam_id"])

        part_path, local_idx = self._exam_index[exam_id]
        f = self._get_hdf5(part_path)
        # tracings shape: (N, 4096, 12) → slice → (4096, 12) → (12, 4096)
        signal = np.array(f["tracings"][local_idx]).T.astype(np.float32)

        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)

        if self.transform is not None:
            signal = self.transform(signal)

        labels = {col: int(row[col]) for col in self.LABEL_COLS if col in row}
        meta_exclude = set(self.LABEL_COLS) | {"exam_id", "trace_file"}
        meta = {k: v for k, v in row.items() if k not in meta_exclude}

        sample = {"signal": signal, "labels": labels, "meta": meta}
        if self.load_error_labels and self._error_df is not None:
            erow = self._error_df.loc[exam_id]
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
        for f in self._hdf5_handles.values():
            try:
                f.close()
            except Exception:
                pass

    # ── Convenience constructors ──────────────────────────────────────
    @classmethod
    def from_patient_split(
        cls,
        root: str | Path = DEFAULT_DATA_DIR,
        split: str = "train",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> "CODE15Dataset":
        """Return a patient-stratified train / val / test split.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            val_fraction: Fraction of patients used for validation.
            test_fraction: Fraction of patients used for testing.
            seed: Random seed for reproducibility.
        """
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")

        df = pd.read_csv(Path(root) / "exams.csv")
        all_patients = np.array(sorted(df["patient_id"].unique()))

        rng = np.random.default_rng(seed)
        rng.shuffle(all_patients)

        n = len(all_patients)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)

        test_patients = all_patients[:n_test]
        val_patients = all_patients[n_test: n_test + n_val]
        train_patients = all_patients[n_test + n_val:]

        mapping = {"train": train_patients, "val": val_patients, "test": test_patients}
        return cls(root, patient_ids=mapping[split].tolist(), **kwargs)
