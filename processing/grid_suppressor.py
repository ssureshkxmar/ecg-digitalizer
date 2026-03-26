"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 3 — Advanced Multi-Strategy Grid Suppression Engine     ║
║  Three independent strategies fused for maximum robustness     ║
╚══════════════════════════════════════════════════════════════════╝

Strategy A: HSV / LAB colour-space segmentation (handles coloured grids)
Strategy B: FFT frequency-domain periodic pattern removal
Strategy C: Morphological line detection & subtraction (handles all grids)
Strategy D: Intensity-based grid detection (light gray lines on white)
Final: Weighted fusion based on detected grid type
"""

import cv2
import numpy as np


# ─── Strategy A: Colour-Space Grid Removal ───────────────────────────────────
def _color_based_removal(colour: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Use HSV + LAB colour spaces to isolate coloured grid lines
    and in-paint them out while preserving the dark waveform.
    """
    h, w = gray.shape
    hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)

    # Red hues (wrap around 0/180)
    mask_r1 = cv2.inRange(hsv, np.array([0, 20, 40]),   np.array([18, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([155, 20, 40]),  np.array([180, 255, 255]))
    # Pink / light red
    mask_pink = cv2.inRange(hsv, np.array([0, 5, 130]),  np.array([25, 130, 255]))
    # Orange / salmon
    mask_orange = cv2.inRange(hsv, np.array([5, 30, 80]), np.array([30, 220, 255]))
    # Green grids
    mask_green = cv2.inRange(hsv, np.array([35, 20, 60]), np.array([85, 255, 255]))
    # Blue grids
    mask_blue = cv2.inRange(hsv, np.array([85, 20, 60]),  np.array([130, 255, 255]))

    grid_mask = mask_r1 | mask_r2 | mask_pink | mask_orange | mask_green | mask_blue
    grid_mask = cv2.dilate(grid_mask,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=1)

    # Protect dark waveform: remove pixels that are very dark
    # (the ECG trace is typically the darkest element)
    dark_mask = gray < 80  # dark waveform pixels
    grid_mask[dark_mask] = 0  # don't remove dark pixels

    if cv2.countNonZero(grid_mask) > 0.005 * h * w:
        result = cv2.inpaint(gray, grid_mask, 5, cv2.INPAINT_TELEA)
        return result

    return gray.copy()


# ─── Strategy B: FFT Frequency-Domain Grid Removal ──────────────────────────
def _fft_grid_removal(gray: np.ndarray) -> np.ndarray:
    """Remove periodic grid pattern via FFT notch filtering."""
    h, w = gray.shape
    dft_h = cv2.getOptimalDFTSize(h)
    dft_w = cv2.getOptimalDFTSize(w)
    padded = np.zeros((dft_h, dft_w), dtype=np.float32)
    padded[:h, :w] = gray.astype(np.float32)

    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))

    mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    mag_log = np.log1p(mag)
    cy, cx = dft_h // 2, dft_w // 2
    notch_mask = np.ones((dft_h, dft_w, 2), dtype=np.float32)

    # Suppress horizontal-axis peaks (vertical grid lines)
    h_profile = mag_log[cy, :]
    h_mean, h_std = np.mean(h_profile), np.std(h_profile)
    for x in range(dft_w):
        if abs(x - cx) > 5 and h_profile[x] > h_mean + 2.0 * h_std:
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    yy, xx = cy + dy, x + dx
                    if 0 <= yy < dft_h and 0 <= xx < dft_w:
                        d = np.sqrt(dy**2 + dx**2)
                        notch_mask[yy, xx] *= max(0, 1 - np.exp(-d**2 / 6))

    # Suppress vertical-axis peaks (horizontal grid lines)
    v_profile = mag_log[:, cx]
    v_mean, v_std = np.mean(v_profile), np.std(v_profile)
    for y in range(dft_h):
        if abs(y - cy) > 5 and v_profile[y] > v_mean + 2.0 * v_std:
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    yy, xx = y + dy, cx + dx
                    if 0 <= yy < dft_h and 0 <= xx < dft_w:
                        d = np.sqrt(dy**2 + dx**2)
                        notch_mask[yy, xx] *= max(0, 1 - np.exp(-d**2 / 6))

    filtered_shift = dft_shift * notch_mask
    filtered = np.fft.ifftshift(filtered_shift, axes=(0, 1))
    result = cv2.idft(filtered)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    result = result[:h, :w]
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


# ─── Strategy C: Morphological Line Detection (Stronger) ────────────────────
def _morphological_grid_removal(gray: np.ndarray) -> np.ndarray:
    """
    Detect grid lines using LONG structuring elements.
    Key: kernels must be longer than any ECG wave feature
    to avoid removing the waveform itself.
    """
    h, w = gray.shape
    inv = cv2.bitwise_not(gray)

    # Horizontal lines — kernel must be very long (>5% of width)
    h_len = max(w // 15, 50)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Vertical lines — kernel must be very long (>5% of height)
    v_len = max(h // 15, 50)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    grid = cv2.add(h_lines, v_lines)
    # Dilate to cover grid line edges
    grid = cv2.dilate(grid,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                      iterations=1)

    cleaned_inv = cv2.subtract(inv, grid)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_inv = cv2.morphologyEx(cleaned_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.bitwise_not(cleaned_inv)


# ─── Strategy D: Intensity-Based Light Grid Removal ──────────────────────────
def _intensity_grid_removal(gray: np.ndarray) -> np.ndarray:
    """
    For ECGs on white paper with light gray grid:
    The waveform is very dark, the grid is light gray, background is white.
    Simply threshold to keep only the darkest elements.
    """
    h, w = gray.shape

    # Compute image statistics
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # If it's a bright image (white paper ECG), use intensity to separate
    if mean_val > 150:
        # Dark trace threshold: keep only pixels significantly darker than background
        threshold = mean_val - 2.0 * std_val
        threshold = max(threshold, 100)
        threshold = min(threshold, 180)

        result = gray.copy()
        # Make light pixels (grid + background) uniformly white
        result[result > threshold] = 255
        return result

    return gray.copy()


# ─── Fusion ──────────────────────────────────────────────────────────────────
def _fuse_results(results, weights):
    total = sum(weights)
    fused = np.zeros_like(results[0], dtype=np.float64)
    for img, w in zip(results, weights):
        fused += img.astype(np.float64) * (w / total)
    return np.clip(fused, 0, 255).astype(np.uint8)


# ─── PUBLIC API ──────────────────────────────────────────────────────────────
def suppress_grid(colour: np.ndarray, gray: np.ndarray) -> dict:
    """
    Execute all grid removal strategies and fuse adaptively.
    """
    # Run all strategies
    colour_cleaned = _color_based_removal(colour, gray)
    fft_cleaned = _fft_grid_removal(gray)
    morph_cleaned = _morphological_grid_removal(gray)
    intensity_cleaned = _intensity_grid_removal(gray)

    # Detect grid type to choose weighting
    hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
    avg_sat = float(np.mean(hsv[:, :, 1]))
    mean_gray = float(np.mean(gray))

    if avg_sat > 25:
        # Coloured grid (red/pink/green) — colour strategy dominates
        weights = [0.45, 0.15, 0.20, 0.20]
    elif mean_gray > 170:
        # White paper with light gray grid — intensity strategy dominates
        weights = [0.10, 0.15, 0.25, 0.50]
    else:
        # Dark/grayscale — morphological + FFT
        weights = [0.15, 0.30, 0.35, 0.20]

    fused = _fuse_results(
        [colour_cleaned, fft_cleaned, morph_cleaned, intensity_cleaned],
        weights)

    return {
        "fused": fused,
        "strategies": {
            "colour": colour_cleaned,
            "fft": fft_cleaned,
            "morphological": morph_cleaned,
        },
        "weights_used": weights,
    }
