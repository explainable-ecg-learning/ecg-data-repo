from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinc_style_dataset import CinCStyleDataset


class ChapmanShaoxingDataset(CinCStyleDataset):
    """PyTorch Dataset for the Chapman-Shaoxing ECG database (PhysioNet).

    45,152 patients · 500 Hz · 10 s · SNOMED-CT labels.

    Files are organised as::

        <root>/WFDBRecords/<XX>/<YY>/<ID>.{hea,mat}

    with nested ``RECORDS`` manifests expanded automatically.

    Args:
        root: Path to the Chapman-Shaoxing root directory (contains
            ``RECORDS`` and ``ConditionNames_SNOMED-CT.csv``).
        load_error_labels: If ``True``, attaches ``sample["error_labels"]`` from
            ``chapman_shaoxing/quality/error_labels.csv``.
        max_error_severity: Optional ``0..3`` cutoff (e.g. ``2`` excludes
            ``"3_severe_error"`` records).
        exclude_error_flags: Optional list of ``err_*`` column names to exclude.
        error_labels_path: Path to the error-labels CSV.
        transform: Optional callable applied to the signal ``(12, seq_len)``.
    """

    PHYSIONET_URL = "https://physionet.org/files/ecg-arrhythmia/1.0.0"
    PACKAGE_DIR = Path(__file__).resolve().parent
    QUALITY_DIR = PACKAGE_DIR / "quality"
    DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
    ERROR_LABELS_CSV = QUALITY_DIR / "error_labels.csv"
    LABEL_CSV = "ConditionNames_SNOMED-CT.csv"
    SAMPLING_RATE = 500

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_DIR,
        load_error_labels: bool = False,
        max_error_severity: Optional[int] = None,
        exclude_error_flags: Optional[list[str]] = None,
        error_labels_path: str | Path = ERROR_LABELS_CSV,
        transform: Optional[Callable] = None,
    ) -> None:
        if max_error_severity is not None and max_error_severity not in (0, 1, 2, 3):
            raise ValueError("max_error_severity must be 0, 1, 2, 3, or None")

        super().__init__(root=root, transform=transform)

        self.load_error_labels = load_error_labels
        self._error_df: Optional[pd.DataFrame] = None
        self._error_flag_cols: list[str] = []

        need_err = load_error_labels or (max_error_severity is not None) or bool(exclude_error_flags)
        if need_err:
            err_path = Path(error_labels_path)
            if not err_path.exists():
                raise FileNotFoundError(
                    f"Error labels CSV not found at {err_path}. "
                    "Run `python chapman_shaoxing/quality/generate_error_labels.py` to generate it."
                )
            self._error_df = pd.read_csv(err_path, index_col="record_id")
            self._error_df["error_tags"] = self._error_df["error_tags"].fillna("")
            self._error_flag_cols = [c for c in self._error_df.columns if c.startswith("err_")]

            if max_error_severity is not None or exclude_error_flags:
                rec_ids = [Path(r).name for r in self._records]
                err_sub = self._error_df.reindex(rec_ids)
                keep = np.ones(len(self._records), dtype=bool)
                if max_error_severity is not None:
                    tier_num = err_sub["severity_tier"].str.slice(0, 1).astype(int).to_numpy()
                    keep &= tier_num <= max_error_severity
                if exclude_error_flags:
                    for col in exclude_error_flags:
                        if col not in err_sub.columns:
                            raise ValueError(f"Unknown error flag '{col}'")
                        keep &= err_sub[col].to_numpy() == 0
                self._records = [r for r, k in zip(self._records, keep) if k]

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        if self.load_error_labels and self._error_df is not None:
            rec_id = Path(sample["meta"]["record_path"]).name
            erow = self._error_df.loc[rec_id]
            tags_str = str(erow["error_tags"])
            sample["error_labels"] = {
                "severity_tier": str(erow["severity_tier"]),
                "recommended_action": str(erow["recommended_action"]),
                "n_error_flags": int(erow["n_error_flags"]),
                "error_tags": tags_str.split("|") if tags_str else [],
                "flags": {c: int(erow[c]) for c in self._error_flag_cols},
            }
        return sample
