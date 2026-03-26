"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 8 — Biomedical Signal Optimization Engine               ║
║  Medical-grade signal processing chain                         ║
╚══════════════════════════════════════════════════════════════════╝

Implements:
  • Wavelet denoising (PyWavelets)
  • Empirical Mode Decomposition (simplified Hilbert-Huang)
  • Adaptive notch filtering (50 Hz + 60 Hz)
  • Baseline wander correction (polynomial fitting)
  • Savitzky-Golay smoothing
  • Butterworth bandpass (0.5–40 Hz)
"""

import numpy as np
from scipy.signal import (
    butter, filtfilt, iirnotch, savgol_filter, medfilt
)

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


# ─── Wavelet Denoising ──────────────────────────────────────────────────────
def _wavelet_denoise(signal: np.ndarray, wavelet: str = 'db6',
                     level: int = 4) -> np.ndarray:
    """
    Wavelet-based ECG denoising using soft thresholding.
    """
    if not HAS_PYWT or len(signal) < 32:
        return signal.copy()

    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    if level < 1:
        return signal.copy()

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise from highest-frequency detail coefficients
    sigma = float(np.median(np.abs(coeffs[-1])) / 0.6745)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Soft thresholding on detail coefficients (keep approximation)
    denoised_coeffs = [coeffs[0]]  # keep approximation
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))

    return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]


# ─── Simplified EMD (Empirical Mode Decomposition) ──────────────────────────
def _simple_emd_baseline(signal: np.ndarray, n_imfs: int = 3) -> np.ndarray:
    """
    Simplified EMD-inspired baseline extraction.
    Uses sifting to extract low-frequency trend.
    """
    if len(signal) < 20:
        return np.zeros_like(signal)

    residual = signal.copy()
    for _ in range(n_imfs):
        # Find local maxima and minima
        from scipy.signal import argrelextrema
        max_idx = argrelextrema(residual, np.greater, order=5)[0]
        min_idx = argrelextrema(residual, np.less, order=5)[0]

        if len(max_idx) < 3 or len(min_idx) < 3:
            break

        # Interpolate upper and lower envelopes
        x = np.arange(len(residual))
        try:
            from scipy.interpolate import CubicSpline
            upper = CubicSpline(max_idx, residual[max_idx],
                                bc_type='natural')(x)
            lower = CubicSpline(min_idx, residual[min_idx],
                                bc_type='natural')(x)
            mean_env = (upper + lower) / 2
            imf = residual - mean_env
            residual = mean_env
        except Exception:
            break

    return residual  # the low-freq baseline


# ─── Polynomial Baseline Correction ─────────────────────────────────────────
def _polynomial_baseline(signal: np.ndarray, order: int = 6) -> np.ndarray:
    """Remove baseline wander using polynomial fit."""
    x = np.arange(len(signal), dtype=np.float64)
    try:
        coeffs = np.polyfit(x, signal, order)
        baseline = np.polyval(coeffs, x)
        return signal - baseline
    except Exception:
        return signal


# ─── Bandpass Filter ─────────────────────────────────────────────────────────
def _bandpass(signal: np.ndarray, fs: float,
              low: float = 0.5, high: float = 40.0) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return signal
    b, a = butter(4, [lo, hi], btype='band')
    padlen = min(len(signal) - 1, 3 * max(len(a), len(b)))
    try:
        return filtfilt(b, a, signal, padlen=padlen)
    except Exception:
        return signal


# ─── Notch Filter ────────────────────────────────────────────────────────────
def _notch(signal: np.ndarray, fs: float, freq: float = 50.0,
           Q: float = 30.0) -> np.ndarray:
    if fs <= 0 or freq >= fs / 2 or len(signal) < 10:
        return signal
    b, a = iirnotch(freq, Q, fs)
    padlen = min(len(signal) - 1, 3 * max(len(a), len(b)))
    try:
        return filtfilt(b, a, signal, padlen=padlen)
    except Exception:
        return signal


# ─── Savitzky-Golay ──────────────────────────────────────────────────────────
def _savgol(signal: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    if len(signal) < window:
        return signal
    win = window if window % 2 == 1 else window + 1
    return savgol_filter(signal, win, poly)


# ─── Median Baseline Removal ────────────────────────────────────────────────
def _median_baseline(signal: np.ndarray, fs: float) -> np.ndarray:
    """Two-pass median filter for baseline wander."""
    if len(signal) < 10:
        return signal
    w1 = int(0.2 * fs)
    w1 = w1 if w1 % 2 == 1 else w1 + 1
    w1 = max(w1, 3)
    ks1 = min(w1, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    b1 = medfilt(signal, kernel_size=ks1)

    w2 = int(0.6 * fs)
    w2 = w2 if w2 % 2 == 1 else w2 + 1
    w2 = max(w2, 3)
    ks2 = min(w2, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    b2 = medfilt(b1, kernel_size=ks2)

    return signal - b2


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════
def optimize_signal(voltage: list, sample_rate: float, config: dict = None) -> dict:
    """
    Full biomedical signal optimization chain with options.
    Returns raw, intermediate, and final processed signals.
    """
    if config is None:
        config = {}

    sig = np.array(voltage, dtype=np.float64)
    fs = float(sample_rate)

    if len(sig) < 20 or fs <= 0:
        return {"raw": sig.tolist(), "filtered": sig.tolist(),
                "stages": {}, "sample_rate": fs}

    stages = {}

    # 1. Wavelet denoising
    if config.get("wavelet_denoise", True):
        s1 = _wavelet_denoise(sig)
    else:
        s1 = sig.copy()
    stages["wavelet"] = s1.tolist()

    # 2. Baseline wander removal (polynomial)
    s2 = _polynomial_baseline(s1)
    stages["polynomial_baseline"] = s2.tolist()

    # 3. Median baseline removal
    s3 = _median_baseline(s2, fs)
    stages["median_baseline"] = s3.tolist()

    # 4. Bandpass 0.5–40 Hz
    s4 = _bandpass(s3, fs, 0.5, 40.0)
    stages["bandpass"] = s4.tolist()

    # 5. Notch Filter
    notch_val = str(config.get("notch_filter", "50"))
    if notch_val == "50":
        s5 = _notch(s4, fs, 50.0)
    elif notch_val == "60":
        s5 = _notch(s4, fs, 60.0)
    else:
        s5 = s4.copy() # Skip notch

    stages["notch"] = s5.tolist()

    # 7. Savitzky-Golay smoothing
    win = min(11, len(s5) if len(s5) % 2 == 1 else len(s5) - 1)
    win = max(win, 5)
    s7 = _savgol(s5, window=win, poly=3)
    stages["savgol"] = s7.tolist()

    return {
        "raw": sig.tolist(),
        "filtered": s7.tolist(),
        "stages": stages,
        "sample_rate": fs,
    }
