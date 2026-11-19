from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinc_style_dataset import CinCStyleDataset


class GeorgiaECGDataset(CinCStyleDataset):
    """PyTorch Dataset for the Georgia 12-Lead ECG database (PhysioNet).

    ~10,000 patients · 500 Hz · 10 s · 27 SNOMED-CT classes.
    Originally released for the PhysioNet/CinC 2020 Challenge.

    Files are organised as::

        <root>/WFDBRecords/<XX>/<YY>/<ID>.{hea,mat}

    with all record paths listed in ``<root>/RECORDS``.

    Args:
        root: Path to the Georgia ECG root directory (contains ``RECORDS``
            and ``ConditionNames_SNOMED-CT.csv``).
        transform: Optional callable applied to the signal ``(12, seq_len)``.
    """

    PHYSIONET_URL = "https://physionet.org/files/challenge-2020/1.0.2/training/georgia"
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
    LABEL_CSV = "ConditionNames_SNOMED-CT.csv"
    SAMPLING_RATE = 500

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_DIR,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=root, transform=transform)
