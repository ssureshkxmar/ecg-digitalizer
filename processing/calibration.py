"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 7 — Adaptive Calibration Engine                         ║
║  Auto-detect grid spacing + validate + calibrate signal        ║
╚══════════════════════════════════════════════════════════════════╝

Detects:
  • Small box size (1 mm) via auto-correlation
  • Validates 5 small = 1 large box
  • Auto-corrects scaling distortion
  • Converts pixel → time (s) and voltage (mV)
"""

import cv2
import numpy as np
from scipy.signal import find_peaks


def _autocorrelation_spacing(profile: np.ndarray, min_lag: int = 5) -> float:
    """Find dominant periodicity using auto-correlation."""
    profile = profile.astype(np.float64)
    profile -= np.mean(profile)
    if np.std(profile) < 1e-6:
        return 0.0

    corr = np.correlate(profile, profile, mode='full')
    corr = corr[len(corr) // 2:]  # positive lags only
    corr = corr / (corr[0] + 1e-10)  # normalise

    peaks, props = find_peaks(corr[min_lag:], height=0.1, distance=min_lag)
    if len(peaks) > 0:
        return float(peaks[0] + min_lag)
    return 0.0


def _detect_grid_spacing(enhanced_gray: np.ndarray) -> dict:
    """
    Multi-axis grid spacing detection.
    Analyses both horizontal and vertical profiles.
    """
    h, w = enhanced_gray.shape

    # Horizontal profile: average across columns → detects horizontal grid lines
    h_profile = np.mean(enhanced_gray, axis=1)
    h_spacing = _autocorrelation_spacing(h_profile)

    # Vertical profile: average across rows → detects vertical grid lines
    v_profile = np.mean(enhanced_gray, axis=0)
    v_spacing = _autocorrelation_spacing(v_profile)

    # Take the most reliable spacing
    spacings = [s for s in [h_spacing, v_spacing] if s > 3]
    if spacings:
        small_box_px = float(np.median(spacings))
    else:
        small_box_px = 20.0  # fallback

    # Validate: 5 small boxes should roughly equal 1 large box
    large_box_px = small_box_px * 5
    # Check if there's a secondary peak near 5× the small spacing
    h_spacing_large = _autocorrelation_spacing(h_profile, min_lag=int(small_box_px * 3))
    v_spacing_large = _autocorrelation_spacing(v_profile, min_lag=int(small_box_px * 3))

    large_spacings = [s for s in [h_spacing_large, v_spacing_large]
                      if abs(s - large_box_px) < large_box_px * 0.4]
    validated = len(large_spacings) > 0

    if large_spacings:
        actual_large = float(np.mean(large_spacings))
        # Refine small box from validated large box
        refined_small = actual_large / 5.0
        if abs(refined_small - small_box_px) < small_box_px * 0.3:
            small_box_px = refined_small

    return {
        "small_box_px": round(small_box_px, 2),
        "large_box_px": round(small_box_px * 5, 2),
        "h_spacing": round(h_spacing, 2),
        "v_spacing": round(v_spacing, 2),
        "five_box_validated": bool(validated),
    }


def calibrate_signal(x_px: np.ndarray, y_px: np.ndarray,
                     enhanced_gray: np.ndarray) -> dict:
    """
    Full calibration: detect grid + convert pixel coordinates to
    physical time (seconds) and voltage (millivolts).

    Standard ECG calibration:
      25 mm/s → 1 small box (1 mm) = 0.04 s
      10 mm/mV → 1 small box (1 mm) = 0.1 mV
    """
    grid = _detect_grid_spacing(enhanced_gray)
    small = grid["small_box_px"]

    if small <= 0:
        small = 20.0

    # Calibration factors
    mm_per_px = 1.0 / small
    time_per_px = 0.04 * mm_per_px   # seconds per pixel
    mv_per_px = 0.1 * mm_per_px      # millivolts per pixel

    # Baseline: median y (isoelectric line)
    baseline_y = float(np.median(y_px)) if len(y_px) > 0 else 0

    # Convert
    time_s = (x_px - x_px[0]) * time_per_px if len(x_px) > 0 else np.array([])
    voltage_mv = (baseline_y - y_px) * mv_per_px if len(y_px) > 0 else np.array([])

    # Sample rate
    if len(time_s) > 1:
        dt = float(np.median(np.diff(time_s)))
        sample_rate = 1.0 / dt if dt > 0 else 250.0
    else:
        sample_rate = 250.0

    return {
        "time_s": time_s.tolist() if isinstance(time_s, np.ndarray) else [],
        "voltage_mv": voltage_mv.tolist() if isinstance(voltage_mv, np.ndarray) else [],
        "baseline_y": baseline_y,
        "sample_rate_hz": round(float(sample_rate), 2),
        "grid": grid,
        "calibration": {
            "mm_per_px": round(mm_per_px, 4),
            "time_per_px_s": round(time_per_px, 6),
            "mv_per_px": round(mv_per_px, 4),
        }
    }
