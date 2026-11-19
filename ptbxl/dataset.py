from __future__ import annotations

import ast
import io
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset


class PTBXLDataset(Dataset):
    """PyTorch Dataset for the PTB-XL ECG dataset.

    Metadata is resolved against ``scp_statements.csv`` to produce a single
    ``diagnostic_class`` and ``diagnostic_subclass`` per record (using the
    highest-weighted diagnostic SCP code).  Non-diagnostic codes and records
    flagged with signal-quality problems can optionally be excluded.

    Each sample returned by ``__getitem__`` is a dict with:
        - ``signal``             : float32 ndarray ``(12, seq_len)``
        - ``scp_codes``          : raw dict of SCP code → likelihood (0–100)
        - ``diagnostic_class``   : str superclass (e.g. ``"NORM"``, ``"MI"``)
        - ``diagnostic_subclass``: str subclass (e.g. ``"IMI"``, ``"LAFB"``)
        - ``extra_labels``       : list[str] of secondary subclasses
        - ``meta``               : remaining CSV columns as a dict

    Args:
        root: Path to the PTB-XL root directory (contains
            ``ptbxl_database.csv`` and ``scp_statements.csv``).
        sampling_rate: ``100`` (default) or ``500`` Hz.
        folds: Stratified folds to include (1–10).  ``None`` keeps all rows.
        filter_noise: If ``True`` (default), records with any signal-quality
            annotation (``static_noise``, ``burst_noise``,
            ``electrodes_problems``) are dropped.
        min_class_fraction: Subclasses whose relative frequency falls below
            this threshold are removed.  Set to ``0.0`` to keep all classes.
        transform: Optional callable applied to the signal array.
    """

    LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    SEGMENTATION_CLASSES = [
        "P-Onset", "(p", "P-Peak", "p)", "P-Offset", "pq",
        "QRS-Onset", "(q", "Q-Peak", "qr", "R-Peak", "rs", "S-Peak", "q)", "QRS-Offset", "qj", "J-point",
        "jt", "T-Onset", "(t", "T-Peak", "t)", "T-Offset", "tp",
    ]
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
    ERROR_LABELS_CSV = Path(__file__).resolve().parent / "error_labels.csv"

    _NOISE_COLS = ["static_noise", "burst_noise", "electrodes_problems"]
    _FOLD_TRAIN = list(range(1, 9))
    _FOLD_VAL = [9]
    _FOLD_TEST = [10]

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_DIR,
        sampling_rate: int = 100,
        folds: Optional[list[int]] = None,
        filter_noise: bool = True,
        min_class_fraction: float = 0.0029,
        load_segmentation: bool = False,
        load_error_labels: bool = False,
        max_error_severity: Optional[int] = None,
        exclude_error_flags: Optional[list[str]] = None,
        error_labels_path: str | Path = ERROR_LABELS_CSV,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        if load_segmentation and sampling_rate != 100:
            raise ValueError("Soft segmentations (Zenodo 7610236) are defined at sampling_rate=100 Hz")
        if max_error_severity is not None and max_error_severity not in (0, 1, 2, 3):
            raise ValueError("max_error_severity must be 0, 1, 2, 3, or None")

        self.root = Path(root)
        self.sampling_rate = sampling_rate
        self.load_segmentation = load_segmentation
        self.load_error_labels = load_error_labels
        self.transform = transform
        self._error_df: Optional[pd.DataFrame] = None
        self._error_flag_cols: list[str] = []

        need_err_table = load_error_labels or (max_error_severity is not None) or bool(exclude_error_flags)
        if need_err_table:
            err_path = Path(error_labels_path)
            if not err_path.exists():
                raise FileNotFoundError(
                    f"Error labels CSV not found at {err_path}. "
                    "Run `python ptbxl/generate_error_labels.py` to generate it."
                )
            self._error_df = pd.read_csv(err_path, index_col="ecg_id")
            self._error_df["error_tags"] = self._error_df["error_tags"].fillna("")
            self._error_flag_cols = [c for c in self._error_df.columns if c.startswith("err_")]
        self._seg_zip_path = self.root / "ptbxl_segmentations_8bit.zip"
        self._seg_zf: Optional[zipfile.ZipFile] = None
        self._seg_member_map: Optional[dict[int, str]] = None
        if self.load_segmentation and not self._seg_zip_path.exists():
            raise FileNotFoundError(
                f"Soft segmentations archive not found at {self._seg_zip_path}. "
                "Run ./ptbxl/download.sh to download Zenodo record 7610236."
            )

        # ── Load metadata ──────────────────────────────────────────────
        df = pd.read_csv(self.root / "ptbxl_database.csv")
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        # ── Load SCP statements (diagnostic codes only) ────────────────
        scp_df = pd.read_csv(self.root / "scp_statements.csv", index_col=0)
        scp_df = scp_df[scp_df["diagnostic"] == 1]

        # ── Drop records with missing age / sex ────────────────────────
        df = df.dropna(subset=["age", "sex"])

        # ── Resolve diagnostic class / subclass ────────────────────────
        df[["diagnostic_subclass", "extra_labels"]] = df["scp_codes"].apply(
            lambda x: pd.Series(self._pick_subclass(x, scp_df))
        )
        df["diagnostic_class"] = df["scp_codes"].apply(
            lambda x: self._aggregate_class(x, scp_df)
        )

        # ── Noise filtering ────────────────────────────────────────────
        if filter_noise:
            noise_cols = [c for c in self._NOISE_COLS if c in df.columns]
            if noise_cols:
                noisy = df[noise_cols].notnull().any(axis=1)
                df = df[~noisy]

        # ── Age sanity check ───────────────────────────────────────────
        df = df[df["age"] <= 100]

        # ── Optional error-label filtering ─────────────────────────────
        if self._error_df is not None and (max_error_severity is not None or exclude_error_flags):
            err_sub = self._error_df.reindex(df["ecg_id"])
            keep_mask = np.ones(len(df), dtype=bool)
            if max_error_severity is not None:
                tier_num = err_sub["severity_tier"].str.slice(0, 1).astype(int).to_numpy()
                keep_mask &= tier_num <= max_error_severity
            if exclude_error_flags:
                for flag_col in exclude_error_flags:
                    if flag_col not in err_sub.columns:
                        raise ValueError(f"Unknown error flag '{flag_col}'")
                    keep_mask &= err_sub[flag_col].to_numpy() == 0
            df = df[keep_mask]

        # ── Fold filter ────────────────────────────────────────────────
        if folds is not None:
            df = df[df["strat_fold"].isin(folds)]

        # ── Rare-class pruning ─────────────────────────────────────────
        if min_class_fraction > 0.0 and len(df) > 0:
            counts = df["diagnostic_subclass"].value_counts()
            freq = counts / len(df)
            keep = freq[freq >= min_class_fraction].index
            df = df[df["diagnostic_subclass"].isin(keep)]

        self._df = df.reset_index(drop=True)
        self._filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

        # Ordered list of unique subclasses (sorted for determinism)
        self.class_names: list[str] = sorted(self._df["diagnostic_subclass"].unique().tolist())

    # ── Convenience constructors ───────────────────────────────────────
    @classmethod
    def train(cls, root: str | Path = DEFAULT_DATA_DIR, **kwargs) -> "PTBXLDataset":
        """Return the training split (folds 1–8)."""
        return cls(root, folds=cls._FOLD_TRAIN, **kwargs)

    @classmethod
    def val(cls, root: str | Path = DEFAULT_DATA_DIR, **kwargs) -> "PTBXLDataset":
        """Return the validation split (fold 9)."""
        return cls(root, folds=cls._FOLD_VAL, **kwargs)

    @classmethod
    def test(cls, root: str | Path = DEFAULT_DATA_DIR, **kwargs) -> "PTBXLDataset":
        """Return the test split (fold 10), noise filtering disabled."""
        kwargs.setdefault("filter_noise", False)
        return cls(root, folds=cls._FOLD_TEST, **kwargs)

    # ── Dataset protocol ──────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.iloc[idx]
        record_path = str(self.root / row[self._filename_col])
        signal, _ = wfdb.rdsamp(record_path)

        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)

        # (seq_len, 12) → (12, seq_len)
        signal = signal.T.astype(np.float32)

        if self.transform is not None:
            signal = self.transform(signal)

        meta_exclude = {"scp_codes", "diagnostic_subclass", "diagnostic_class", "extra_labels",
                        self._filename_col}
        sample = {
            "signal": signal,
            "scp_codes": row["scp_codes"],
            "diagnostic_class": row["diagnostic_class"],
            "diagnostic_subclass": row["diagnostic_subclass"],
            "extra_labels": row["extra_labels"],
            "meta": {k: v for k, v in row.items() if k not in meta_exclude},
        }
        if self.load_segmentation:
            sample["segmentation"] = self._load_segmentation(int(row["ecg_id"]))
        if self.load_error_labels and self._error_df is not None:
            erow = self._error_df.loc[int(row["ecg_id"])]
            tags_str = str(erow["error_tags"])
            sample["error_labels"] = {
                "severity_tier": str(erow["severity_tier"]),
                "recommended_action": str(erow["recommended_action"]),
                "n_error_flags": int(erow["n_error_flags"]),
                "n_annotated_noisy_leads": int(erow["n_annotated_noisy_leads"]),
                "error_tags": tags_str.split("|") if tags_str else [],
                "flags": {c: int(erow[c]) for c in self._error_flag_cols},
            }
        return sample

    def _load_segmentation(self, ecg_id: int) -> np.ndarray:
        """Load normalized (12, 24, 1000) float32 soft segmentation for ecg_id."""
        if self._seg_zf is None or self._seg_member_map is None:
            self._seg_zf = zipfile.ZipFile(self._seg_zip_path, "r")
            self._seg_member_map = {}
            for info in self._seg_zf.infolist():
                if not info.is_dir():
                    m = re.match(r"^0*(\d+)(?:\..+)?$", Path(info.filename).name)
                    if m:
                        self._seg_member_map[int(m.group(1))] = info.filename
        raw = self._seg_zf.read(self._seg_member_map[ecg_id])
        # Stored as (1000, 12, 24) uint8 → renormalize simplex → (12, 24, 1000) float32
        u8 = np.asarray(np.load(io.BytesIO(raw), allow_pickle=True), dtype=np.float32)
        probs = u8 / np.maximum(u8.sum(axis=-1, keepdims=True), 1.0)
        return np.transpose(probs, (1, 2, 0))

    # ── Label helpers ─────────────────────────────────────────────────
    def _aggregate_class(self, scp_dict: dict, scp_df: pd.DataFrame) -> str:
        """Map SCP codes to their diagnostic superclass."""
        labels = sorted({
            scp_df.loc[code, "diagnostic_class"]
            for code in scp_dict
            if code in scp_df.index
        })
        return labels[0] if labels else "NORM"

    @staticmethod
    def _normalize_subclass(subclass: str) -> str:
        if subclass in {"ISCA", "ISCI", "ISC_"}:
            return "ISC"
        if subclass == "LMI":
            return "IMI"
        return subclass

    def _pick_subclass(self, scp_dict: dict, scp_df: pd.DataFrame) -> tuple[str, list[str]]:
        """Select the single highest-weighted diagnostic subclass."""
        candidates: list[tuple[str, float]] = []
        total_weight = 0.0

        for code, weight in scp_dict.items():
            if code not in scp_df.index:
                continue
            subclass = self._normalize_subclass(scp_df.loc[code, "diagnostic_subclass"])
            candidates.append((subclass, float(weight)))
            total_weight += float(weight)

        if not candidates:
            return "NORM", []

        best_by_subclass: dict[str, float] = {}
        for subclass, weight in candidates:
            best_by_subclass[subclass] = max(best_by_subclass.get(subclass, 0.0), weight)

        best = max(best_by_subclass.items(), key=lambda item: (item[1], item[0]))[0]

        extra: list[str] = []
        if total_weight > 100.0:
            extra = sorted(s for s in best_by_subclass if s != best)

        return best, extra
