"""
verify_datasets.py — smoke-test all ECG dataset classes.

For each dataset whose data directory exists, the script:
  1. Instantiates the dataset class (all splits where applicable).
  2. Checks that __len__() > 0.
  3. Loads the first sample and validates:
       - signal shape (12, seq_len), dtype float32, no NaNs
       - all expected dict keys are present
       - signal values are within a plausible mV range
  4. Randomly samples 10 additional records and checks signal shape.

Run from the repo root:
    python verify_datasets.py

Exit code 0 → all present datasets passed.
Exit code 1 → at least one failure.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent

_N_RANDOM_CHECKS = 10
_MAX_MV = 50.0   # plausible upper bound for ECG amplitude in mV


def check_signal(signal: np.ndarray, expected_leads: int = 12) -> list[str]:
    """Return a list of error strings; empty list means OK."""
    errors = []
    if not isinstance(signal, np.ndarray):
        errors.append(f"signal is {type(signal)}, expected ndarray")
        return errors
    if signal.ndim != 2:
        errors.append(f"signal.ndim={signal.ndim}, expected 2")
        return errors
    if signal.shape[0] != expected_leads:
        errors.append(f"signal.shape={signal.shape}, expected ({expected_leads}, seq_len)")
    if signal.shape[1] == 0:
        errors.append("signal has zero length in time dimension")
    if signal.dtype != np.float32:
        errors.append(f"signal.dtype={signal.dtype}, expected float32")
    if np.isnan(signal).any():
        errors.append("signal contains NaN values")
    if np.isinf(signal).any():
        errors.append("signal contains Inf values")
    abs_max = np.abs(signal).max()
    if abs_max > _MAX_MV:
        errors.append(f"signal max abs value {abs_max:.1f} exceeds {_MAX_MV} mV — possible unit error")
    return errors


def _assert_signal(signal: np.ndarray, context: str = "") -> None:
    errors = check_signal(signal)
    if errors:
        raise AssertionError(f"{context}: " + "; ".join(errors))


def _assert_keys(sample: dict, required: set, context: str = "") -> None:
    missing = required - sample.keys()
    if missing:
        raise AssertionError(f"{context}: missing keys {missing}")


def _random_check(ds, n: int = _N_RANDOM_CHECKS, signal_key: str = "signal") -> None:
    """Load n random samples and verify signal shape."""
    indices = np.random.default_rng(0).integers(0, len(ds), size=min(n, len(ds)))
    for i in indices:
        sample = ds[int(i)]
        _assert_signal(sample[signal_key], context=f"random idx {i}")


def run_check(name: str, fn) -> bool:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"  RESULT: PASSED")
        return True
    except Exception:
        print(f"  RESULT: FAILED")
        traceback.print_exc()
        return False


# ── PTB-XL ────────────────────────────────────────────────────────────────────
def check_ptbxl():
    data_dir = ROOT / "ptbxl" / "data"
    if not data_dir.exists():
        print(f"  SKIP: {data_dir} not found")
        return

    from ptbxl import PTBXLDataset

    required_keys = {"signal", "scp_codes", "diagnostic_class", "diagnostic_subclass", "extra_labels", "meta"}
    for split_name, ds in [
        ("train", PTBXLDataset.train(data_dir)),
        ("val",   PTBXLDataset.val(data_dir)),
        ("test",  PTBXLDataset.test(data_dir)),
    ]:
        assert len(ds) > 0, f"split '{split_name}' is empty"
        sample = ds[0]
        _assert_keys(sample, required_keys, context=split_name)
        _assert_signal(sample["signal"], context=f"{split_name}[0]")
        assert isinstance(sample["diagnostic_class"], str), "diagnostic_class must be str"
        assert isinstance(sample["diagnostic_subclass"], str), "diagnostic_subclass must be str"
        assert isinstance(sample["scp_codes"], dict), "scp_codes must be dict"
        _random_check(ds)
        print(f"  [{split_name:5s}] len={len(ds):6d} | signal {sample['signal'].shape} | "
              f"class={sample['diagnostic_class']} | sub={sample['diagnostic_subclass']}")

    print(f"  class_names ({len(ds.class_names)}): {ds.class_names[:5]} ...")


# ── MIMIC-IV-ECG ──────────────────────────────────────────────────────────────
def check_mimic_iv_ecg():
    data_dir = ROOT / "mimic_iv_ecg" / "data"
    if not data_dir.exists():
        print(f"  SKIP: {data_dir} not found")
        return

    from mimic_iv_ecg import MIMICIVECGDataset

    required_keys = {"signal", "subject_id", "study_id", "meta"}
    for split_name, ds in [
        ("train", MIMICIVECGDataset.from_subject_split(data_dir, split="train")),
        ("val",   MIMICIVECGDataset.from_subject_split(data_dir, split="val")),
        ("test",  MIMICIVECGDataset.from_subject_split(data_dir, split="test")),
    ]:
        assert len(ds) > 0, f"split '{split_name}' is empty"
        sample = ds[0]
        _assert_keys(sample, required_keys, context=split_name)
        _assert_signal(sample["signal"], context=f"{split_name}[0]")
        assert isinstance(sample["subject_id"], int), "subject_id must be int"
        assert isinstance(sample["study_id"], int), "study_id must be int"
        assert sample["signal"].shape == (12, 5000), \
            f"expected shape (12, 5000), got {sample['signal'].shape}"
        _random_check(ds)
        print(f"  [{split_name:5s}] len={len(ds):6d} | signal {sample['signal'].shape} | "
              f"subject_id={sample['subject_id']}")


# ── Chapman-Shaoxing ──────────────────────────────────────────────────────────
def check_chapman_shaoxing():
    data_dir = ROOT / "chapman_shaoxing" / "data"
    if not data_dir.exists():
        print(f"  SKIP: {data_dir} not found")
        return

    from chapman_shaoxing import ChapmanShaoxingDataset

    required_keys = {"signal", "snomed_codes", "label_names", "meta"}
    ds = ChapmanShaoxingDataset(data_dir)
    assert len(ds) > 0, "dataset is empty"
    sample = ds[0]
    _assert_keys(sample, required_keys, context="ds[0]")
    _assert_signal(sample["signal"], context="ds[0]")
    assert isinstance(sample["snomed_codes"], list), "snomed_codes must be list"
    assert isinstance(sample["label_names"], list), "label_names must be list"
    assert len(sample["snomed_codes"]) > 0, "snomed_codes is empty"
    _random_check(ds)
    print(f"  len={len(ds):6d} | signal {sample['signal'].shape} | "
          f"labels={sample['label_names']}")


# ── Georgia 12-Lead ECG ───────────────────────────────────────────────────────
def check_georgia_ecg():
    data_dir = ROOT / "georgia_ecg" / "data"
    if not data_dir.exists():
        print(f"  SKIP: {data_dir} not found")
        return

    from georgia_ecg import GeorgiaECGDataset

    required_keys = {"signal", "snomed_codes", "label_names", "meta"}
    ds = GeorgiaECGDataset(data_dir)
    assert len(ds) > 0, "dataset is empty"
    sample = ds[0]
    _assert_keys(sample, required_keys, context="ds[0]")
    _assert_signal(sample["signal"], context="ds[0]")
    assert isinstance(sample["snomed_codes"], list), "snomed_codes must be list"
    assert isinstance(sample["label_names"], list), "label_names must be list"
    assert len(sample["snomed_codes"]) > 0, "snomed_codes is empty"
    _random_check(ds)
    print(f"  len={len(ds):6d} | signal {sample['signal'].shape} | "
          f"labels={sample['label_names']}")


# ── CODE-15% ──────────────────────────────────────────────────────────────────
def check_code15():
    data_dir = ROOT / "code15" / "data"
    if not data_dir.exists():
        print(f"  SKIP: {data_dir} not found")
        return

    from code15 import CODE15Dataset

    required_keys = {"signal", "labels", "meta"}
    for split_name, ds in [
        ("train", CODE15Dataset.from_patient_split(data_dir, split="train")),
        ("val",   CODE15Dataset.from_patient_split(data_dir, split="val")),
        ("test",  CODE15Dataset.from_patient_split(data_dir, split="test")),
    ]:
        assert len(ds) > 0, f"split '{split_name}' is empty"
        sample = ds[0]
        _assert_keys(sample, required_keys, context=split_name)
        _assert_signal(sample["signal"], context=f"{split_name}[0]")
        assert isinstance(sample["labels"], dict), "labels must be dict"
        assert set(sample["labels"].keys()) == {"1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"}, \
            f"unexpected label keys: {set(sample['labels'].keys())}"
        assert sample["signal"].shape == (12, 4096), \
            f"expected shape (12, 4096), got {sample['signal'].shape}"
        _random_check(ds)
        print(f"  [{split_name:5s}] len={len(ds):6d} | signal {sample['signal'].shape} | "
              f"labels={sample['labels']}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {
        "PTB-XL":            run_check("PTB-XL",            check_ptbxl),
        "MIMIC-IV-ECG":      run_check("MIMIC-IV-ECG",      check_mimic_iv_ecg),
        "Chapman-Shaoxing":  run_check("Chapman-Shaoxing",  check_chapman_shaoxing),
        "Georgia 12-Lead":   run_check("Georgia 12-Lead",   check_georgia_ecg),
        "CODE-15%":          run_check("CODE-15%",          check_code15),
    }

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    n_passed = n_skipped = n_failed = 0
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
            n_skipped += 1
        elif passed:
            status = "PASSED"
            n_passed += 1
        else:
            status = "FAILED"
            n_failed += 1
        print(f"  {name:<25} {status}")
    print(f"\n  {n_passed} passed, {n_skipped} skipped, {n_failed} failed")

    if n_failed > 0:
        sys.exit(1)
