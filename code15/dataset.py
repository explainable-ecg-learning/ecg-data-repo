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
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_DIR,
        patient_ids: Optional[list[int]] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        df = pd.read_csv(self.root / "exams.csv")
        df.columns = df.columns.str.strip()

        if patient_ids is not None:
            df = df[df["patient_id"].isin(patient_ids)]

        self._df = df.reset_index(drop=True)

        # Build a lookup: exam_id → (part_file_path, index_within_file)
        self._exam_index: dict[int, tuple[Path, int]] = {}
        self._hdf5_handles: dict[Path, h5py.File] = {}
        self._build_index()

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

        return {"signal": signal, "labels": labels, "meta": meta}

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
