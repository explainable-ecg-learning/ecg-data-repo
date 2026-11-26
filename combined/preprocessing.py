"""Unified 12-lead ECG preprocessing, resampling, denoising, and smoothing pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gcd
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, savgol_filter, sosfiltfilt


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for harmonizing multi-source 12-lead ECG waveforms."""

    target_fs: int = 500
    target_duration_s: float = 10.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    bandpass_order: int = 4
    notch_freqs_hz: tuple[float, ...] = (50.0, 60.0)
    notch_q: float = 30.0
    smooth: bool = True
    savgol_window_ms: float = 14.0  # ~7 samples at 500 Hz
    savgol_polyorder: int = 3
    normalize: str = "none"  # "none" (calibrated mV) | "zscore" (per-lead)
    clip_mv: Optional[float] = 20.0
    min_lead_std: float = 1e-4

    @property
    def target_length(self) -> int:
        return int(round(self.target_fs * self.target_duration_s))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notch_freqs_hz"] = list(self.notch_freqs_hz)
        d["target_length"] = self.target_length
        return d


@lru_cache(maxsize=16)
def _cached_bandpass_sos(low_hz: float, high_hz: float, order: int, fs: float) -> Optional[np.ndarray]:
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    if 0.0 < low < high < 1.0:
        return butter(order, [low, high], btype="bandpass", output="sos")
    return None


@lru_cache(maxsize=32)
def _cached_notch_ba(f0: float, q: float, fs: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
    nyq = 0.5 * fs
    if 0.0 < f0 < nyq:
        return iirnotch(w0=f0, Q=q, fs=fs)
    return None


def _sanitize_nans(signal: np.ndarray) -> np.ndarray:
    """Replace isolated NaN/Inf values per lead via linear interpolation."""
    if np.isfinite(signal).all():
        return signal
    out = signal.copy()
    t = np.arange(out.shape[1])
    for i in range(out.shape[0]):
        finite = np.isfinite(out[i])
        if not finite.any():
            out[i] = 0.0
        elif not finite.all():
            out[i, ~finite] = np.interp(t[~finite], t[finite], out[i, finite])
    return out


def _trim_zero_padding(signal: np.ndarray, min_len: int = 100) -> np.ndarray:
    """Strip symmetric leading/trailing all-zero padding (e.g. CODE-15%)."""
    nonzero = np.where(np.any(signal != 0.0, axis=0))[0]
    if len(nonzero) >= min_len:
        return signal[:, nonzero[0] : nonzero[-1] + 1]
    return signal


def _resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Polyphase anti-aliased resampling along the time axis (axis=1)."""
    if orig_fs == target_fs:
        return signal
    g = gcd(int(orig_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(orig_fs) // g
    return resample_poly(signal, up=up, down=down, axis=1).astype(np.float64, copy=False)


def _fix_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop if longer than target_len; symmetrically zero-pad if shorter."""
    cur_len = signal.shape[1]
    if cur_len == target_len:
        return signal
    if cur_len > target_len:
        start = (cur_len - target_len) // 2
        return signal[:, start : start + target_len]
    pad_total = target_len - cur_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0)


def preprocess_ecg(
    signal: np.ndarray,
    orig_fs: int,
    cfg: PreprocessConfig = PreprocessConfig(),
    trim_zero_pad: bool = False,
) -> Optional[np.ndarray]:
    """Clean, resample, denoise, smooth, and standardize a (12, T) ECG array.

    Returns a float32 array of shape ``(12, cfg.target_length)``, or ``None`` if
    the signal fails post-processing quality checks (flatline / non-finite).
    """
    if signal.ndim != 2 or signal.shape[0] != 12 or signal.shape[1] < 100:
        return None

    x = signal.astype(np.float64, copy=True)
    x = _sanitize_nans(x)

    if trim_zero_pad:
        x = _trim_zero_padding(x, min_len=max(100, orig_fs))

    # Remove per-lead DC offset before filtering to avoid edge transients
    x -= np.mean(x, axis=1, keepdims=True)

    # 1. Resample to unified sampling rate (e.g. 400 Hz -> 500 Hz)
    x = _resample_signal(x, orig_fs=orig_fs, target_fs=cfg.target_fs)
    fs = float(cfg.target_fs)

    # 2. Zero-phase bandpass filter (baseline wander removal + HF cutoff)
    sos = _cached_bandpass_sos(cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order, fs)
    if sos is not None and x.shape[1] > 3 * (cfg.bandpass_order * 2 + 1):
        x = sosfiltfilt(sos, x, axis=1)

    # 3. Dual mains-hum notch denoising (50 Hz Europe/China + 60 Hz US/Brazil)
    for f0 in cfg.notch_freqs_hz:
        ba = _cached_notch_ba(f0, cfg.notch_q, fs)
        if ba is not None and x.shape[1] > 30:
            x = filtfilt(ba[0], ba[1], x, axis=1)

    # 4. Zero-phase Savitzky-Golay smoothing (suppresses residual EMG noise while preserving QRS peaks)
    if cfg.smooth:
        win = int(round(cfg.savgol_window_ms * fs / 1000.0))
        if win % 2 == 0:
            win += 1
        win = max(win, cfg.savgol_polyorder + 2)
        if win % 2 == 0:
            win += 1
        if x.shape[1] > win:
            x = savgol_filter(x, window_length=win, polyorder=cfg.savgol_polyorder, axis=1)

    # 5. Standardize temporal length to (12, target_length)
    x = _fix_length(x, cfg.target_length)

    # 6. Safety clip extreme spikes
    if cfg.clip_mv is not None:
        x = np.clip(x, -cfg.clip_mv, cfg.clip_mv)

    # 7. Post-processing quality gate (reject non-finite or dead/flatline leads)
    if not np.isfinite(x).all():
        return None
    lead_stds = np.std(x, axis=1)
    if np.any(lead_stds < cfg.min_lead_std):
        return None

    # 8. Optional per-lead z-score normalization
    if cfg.normalize == "zscore":
        x = (x - np.mean(x, axis=1, keepdims=True)) / (lead_stds[:, None] + 1e-8)
    elif cfg.normalize != "none":
        raise ValueError(f"Unknown normalize mode '{cfg.normalize}' (expected 'none' or 'zscore')")

    return x.astype(np.float32)
