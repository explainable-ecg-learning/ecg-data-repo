from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset


class MIMICIVECGDataset(Dataset):
    """PyTorch Dataset for the MIMIC-IV-ECG dataset (v1.0).

    ~800,000 ten-second 12-lead diagnostic ECGs sampled at 500 Hz from
    ~160,000 unique patients at BIDMC (2008–2019).

    Files are structured as::

        <root>/files/pNNNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ.{hea,dat}

    Metadata is sourced from three CSV files in the root directory:

    * ``record_list.csv``                 – one row per study (always loaded)
    * ``machine_measurements.csv``        – machine summary measures + report
      lines; merged on ``study_id`` when ``load_measurements=True``
    * ``waveform_note_links.csv``         – links to cardiologist notes in
      MIMIC-IV-Note; merged on ``study_id`` when ``load_note_links=True``

    Each sample returned by ``__getitem__`` is a dict with:

    * ``signal``     : float32 ndarray ``(12, 5000)``  (12 leads × 10 s × 500 Hz)
    * ``subject_id`` : int
    * ``study_id``   : int
    * ``meta``       : remaining metadata columns as a dict

    Args:
        root: Path to the MIMIC-IV-ECG root directory (contains
            ``record_list.csv``).
        subject_ids: If given, only records belonging to these subject IDs are
            included.  Useful for creating patient-level train/val/test splits.
        load_measurements: Merge ``machine_measurements.csv`` into metadata.
        load_note_links: Merge ``waveform_note_links.csv`` into metadata.
        transform: Optional callable applied to the signal array.
    """

    LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    SAMPLING_RATE = 500
    SIGNAL_LENGTH = 5000  # 10 s × 500 Hz
    PACKAGE_DIR = Path(__file__).resolve().parent
    QUALITY_DIR = PACKAGE_DIR / "quality"
    DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
    ERROR_LABELS_CSV = QUALITY_DIR / "error_labels.csv.gz"
    _MEAS_SENTINEL_COLS = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
        "p_axis", "qrs_axis", "t_axis",
    ]

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_DIR,
        subject_ids: Optional[list[int]] = None,
        load_measurements: bool = True,
        load_note_links: bool = False,
        clean_sentinels: bool = False,
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
                    "Run `python mimic_iv_ecg/quality/generate_error_labels.py` to generate it."
                )
            self._error_df = pd.read_csv(err_path, index_col="study_id")
            self._error_df["error_tags"] = self._error_df["error_tags"].fillna("")
            self._error_flag_cols = [c for c in self._error_df.columns if c.startswith("err_")]

        # ── record_list.csv ───────────────────────────────────────────
        df = pd.read_csv(self.root / "record_list.csv")
        # Normalise column names (lowercase, strip spaces)
        df.columns = df.columns.str.strip().str.lower()

        # ── optional subject filter ───────────────────────────────────
        if subject_ids is not None:
            df = df[df["subject_id"].isin(subject_ids)]

        # ── machine_measurements.csv ──────────────────────────────────
        if load_measurements:
            meas_path = self.root / "machine_measurements.csv"
            if meas_path.exists():
                meas = pd.read_csv(meas_path, low_memory=False)
                meas.columns = meas.columns.str.strip().str.lower()
                if clean_sentinels:
                    for col in self._MEAS_SENTINEL_COLS:
                        if col in meas.columns:
                            meas[col] = meas[col].replace(29999, np.nan)
                # Drop columns that are already in record_list to avoid duplicates
                overlap = {"subject_id", "study_id", "path"} & set(meas.columns)
                drop_cols = list(overlap - {"subject_id", "study_id"})
                meas = meas.drop(columns=drop_cols, errors="ignore")
                df = df.merge(meas, on=["subject_id", "study_id"], how="left")

        # ── waveform_note_links.csv ───────────────────────────────────
        if load_note_links:
            links_path = self.root / "waveform_note_links.csv"
            if links_path.exists():
                links = pd.read_csv(links_path)
                links.columns = links.columns.str.strip().str.lower()
                # Deduplicate multi-note studies so merging preserves 1 row per study_id
                links = links.drop_duplicates(subset=["subject_id", "study_id"], keep="first")
                drop_cols = list(({"subject_id", "study_id", "path"} & set(links.columns)) - {"subject_id", "study_id"})
                links = links.drop(columns=drop_cols, errors="ignore")
                df = df.merge(links, on=["subject_id", "study_id"], how="left")

        # ── optional error-label filtering ────────────────────────────
        if self._error_df is not None and (max_error_severity is not None or exclude_error_flags):
            err_sub = self._error_df.reindex(df["study_id"])
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

    # ── Convenience constructors ──────────────────────────────────────
    @classmethod
    def from_subject_split(
        cls,
        root: str | Path = DEFAULT_DATA_DIR,
        split: str = "train",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> "MIMICIVECGDataset":
        """Return a subject-stratified train / val / test split.

        Subjects (not records) are randomly partitioned to avoid data leakage.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            val_fraction: Fraction of subjects used for validation.
            test_fraction: Fraction of subjects used for testing.
            seed: Random seed for reproducibility.
        """
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Load only record_list to get subject IDs cheaply
        rl = pd.read_csv(Path(root) / "record_list.csv")
        rl.columns = rl.columns.str.strip().str.lower()
        all_subjects = np.array(sorted(rl["subject_id"].unique()))

        rng = np.random.default_rng(seed)
        rng.shuffle(all_subjects)

        n = len(all_subjects)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)

        test_subjects = all_subjects[:n_test]
        val_subjects = all_subjects[n_test: n_test + n_val]
        train_subjects = all_subjects[n_test + n_val:]

        mapping = {"train": train_subjects, "val": val_subjects, "test": test_subjects}
        return cls(root, subject_ids=mapping[split].tolist(), **kwargs)

    # ── Dataset protocol ──────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.iloc[idx]

        # The 'path' column stores the relative path without extension, e.g.
        # "files/p1000/p10001725/s41420867/41420867"
        record_path = str(self.root / row["path"])
        signal, _ = wfdb.rdsamp(record_path)

        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)

        # (5000, 12) → (12, 5000)
        signal = signal.T.astype(np.float32)

        if self.transform is not None:
            signal = self.transform(signal)

        meta_exclude = {"subject_id", "study_id", "path"}
        sample = {
            "signal": signal,
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "meta": {k: v for k, v in row.items() if k not in meta_exclude},
        }
        if self.load_error_labels and self._error_df is not None:
            erow = self._error_df.loc[int(row["study_id"])]
            tags_str = str(erow["error_tags"])
            sample["error_labels"] = {
                "severity_tier": str(erow["severity_tier"]),
                "recommended_action": str(erow["recommended_action"]),
                "n_error_flags": int(erow["n_error_flags"]),
                "error_tags": tags_str.split("|") if tags_str else [],
                "flags": {c: int(erow[c]) for c in self._error_flag_cols},
            }
        return sample
