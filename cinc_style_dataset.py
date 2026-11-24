from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset


class CinCStyleDataset(Dataset):
    """Base class for PhysioNet/CinC-challenge-style 12-lead ECG datasets.

    These datasets share the same file layout:

    * Records listed in a ``RECORDS`` file (one relative path per line,
      no extension).
    * Each record is a pair of ``<name>.hea`` and ``<name>.mat`` files.
    * Diagnostic labels are SNOMED-CT codes embedded in the ``.hea`` header
      on a line of the form ``#Dx: <code1>,<code2>,...``
    * A mapping CSV (``ConditionNames_SNOMED-CT.csv`` or equivalent) translates
      codes to human-readable names.

    Subclasses set ``PHYSIONET_URL``, ``DEFAULT_DATA_DIR``, and
    ``LABEL_CSV`` to specialise behaviour.

    Each sample returned by ``__getitem__`` is a dict with:

    * ``signal``      : float32 ndarray ``(12, seq_len)``
    * ``snomed_codes``: list[str] of SNOMED-CT codes
    * ``label_names`` : list[str] of human-readable condition names
    * ``meta``        : dict with ``age``, ``sex``, ``record_path``
    """

    LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Override in subclasses
    PHYSIONET_URL: str = ""
    LABEL_CSV: str = "ConditionNames_SNOMED-CT.csv"

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        # ── load record list ───────────────────────────────────────────
        records_file = self.root / "RECORDS"
        with open(records_file) as f:
            self._records: list[str] = [line.strip() for line in f if line.strip()]

        # ── load SNOMED-CT code → name mapping ─────────────────────────
        label_csv = self.root / self.LABEL_CSV
        if label_csv.exists():
            df = pd.read_csv(label_csv)
            # Normalise: expect columns 'Snomed_CT' / 'Abbreviation' or similar
            df.columns = df.columns.str.strip()
            code_col = next((c for c in df.columns if "snomed" in c.lower()), df.columns[0])
            name_col = next((c for c in df.columns if "abbrev" in c.lower() or "name" in c.lower()), df.columns[1])
            self._code_to_name: dict[str, str] = dict(
                zip(df[code_col].astype(str), df[name_col].astype(str))
            )
        else:
            self._code_to_name = {}

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        rel_path = self._records[idx]
        record_path = str(self.root / rel_path)
        signal, _ = wfdb.rdsamp(record_path)

        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)

        # (seq_len, 12) → (12, seq_len)
        signal = signal.T.astype(np.float32)

        if self.transform is not None:
            signal = self.transform(signal)

        snomed_codes, age, sex = self._parse_header(record_path + ".hea")
        label_names = [self._code_to_name.get(c, c) for c in snomed_codes]

        return {
            "signal": signal,
            "snomed_codes": snomed_codes,
            "label_names": label_names,
            "meta": {"age": age, "sex": sex, "record_path": rel_path},
        }

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _parse_header(hea_path: str) -> tuple[list[str], Optional[float], Optional[str]]:
        """Extract SNOMED-CT Dx codes, age, and sex from a .hea file."""
        snomed_codes: list[str] = []
        age: Optional[float] = None
        sex: Optional[str] = None

        with open(hea_path) as f:
            for line in f:
                line = line.strip()
                m_dx = re.search(r"#\s*Dx:\s*(.+)", line, re.IGNORECASE)
                if m_dx:
                    snomed_codes = [c.strip() for c in m_dx.group(1).split(",") if c.strip()]
                m_age = re.search(r"#\s*Age:\s*(\d+\.?\d*)", line, re.IGNORECASE)
                if m_age:
                    age = float(m_age.group(1))
                m_sex = re.search(r"#\s*Sex:\s*(\w+)", line, re.IGNORECASE)
                if m_sex:
                    sex = m_sex.group(1)

        return snomed_codes, age, sex

    @property
    def all_snomed_codes(self) -> list[str]:
        """Sorted list of all unique SNOMED-CT codes present in the dataset."""
        codes: set[str] = set()
        for rec in self._records:
            hea = str(self.root / rec) + ".hea"
            c, _, _ = self._parse_header(hea)
            codes.update(c)
        return sorted(codes)
