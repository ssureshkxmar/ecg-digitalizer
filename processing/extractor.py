"""
Medical-Grade ECG Image Processing Pipeline
=============================================
Stages:
  1. Quality Assessment
  2. Pre-processing (Grayscale, Denoise, CLAHE, Perspective)
  3. Grid Removal (HSV + Morphological)
  4. Wave Isolation (Adaptive Threshold, Morphological Cleaning, Skeletonization)
  5. Waveform Digitization (Baseline detection, Pixel→Voltage/Time calibration)
"""

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. QUALITY ASSESSMENT
# ---------------------------------------------------------------------------
def assess_quality(image: np.ndarray) -> dict:
    """Basic quality metrics for the input image."""
    h, w = image.shape[:2]
    dpi_est = max(h, w) / 10.0  # rough estimate assuming ~10-inch print
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # sharpness
    return {
        "width": int(w),
        "height": int(h),
        "estimated_dpi": int(round(dpi_est)),
        "sharpness": round(float(laplacian_var), 2),
        "is_sharp": bool(laplacian_var > 50),
    }


# ---------------------------------------------------------------------------
# 2. PRE-PROCESSING
# ---------------------------------------------------------------------------
def preprocess(image: np.ndarray) -> tuple:
    """
    Returns (preprocessed_gray, contrast_enhanced) and the resized colour copy.
    """
    # 2a. Resize large images
    h, w = image.shape[:2]
    max_dim = 2400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    colour = image.copy()

    # 2b. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2c. Edge-preserving denoising – Bilateral Filter
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 2d. CLAHE – Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return colour, gray, denoised, enhanced


# ---------------------------------------------------------------------------
# 3. GRID REMOVAL
# ---------------------------------------------------------------------------
def remove_grid(colour_img: np.ndarray, enhanced_gray: np.ndarray) -> np.ndarray:
    """
    Multi-strategy grid removal:
      A) HSV colour masking – remove coloured grids (red / pink / orange)
      B) Morphological line removal – remove thin horizontal & vertical lines
    Returns a clean grayscale image with grid suppressed.
    """
    h, w = enhanced_gray.shape[:2]
    clean = enhanced_gray.copy()

    # --- Strategy A: HSV colour masking for red/pink grids ---
    hsv = cv2.cvtColor(colour_img, cv2.COLOR_BGR2HSV)

    # Red hue wraps around 0/180 in OpenCV HSV
    mask1 = cv2.inRange(hsv, np.array([0, 30, 50]),   np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 30, 50]),  np.array([180, 255, 255]))
    # Pink / light grid
    mask3 = cv2.inRange(hsv, np.array([0, 10, 150]),   np.array([20, 100, 255]))
    # Orange / salmon grid
    mask4 = cv2.inRange(hsv, np.array([5, 40, 100]),   np.array([25, 200, 255]))

    colour_grid_mask = mask1 | mask2 | mask3 | mask4
    colour_grid_mask = cv2.dilate(colour_grid_mask,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                  iterations=1)

    # In-paint the grid areas with local background
    if cv2.countNonZero(colour_grid_mask) > 0.01 * h * w:
        # Significant coloured grid detected → in-paint
        clean = cv2.inpaint(enhanced_gray, colour_grid_mask, 3, cv2.INPAINT_TELEA)

    # --- Strategy B: Morphological line removal for remaining thin lines ---
    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 25), 1))
    h_lines = cv2.morphologyEx(
        cv2.bitwise_not(clean), cv2.MORPH_OPEN, h_kernel, iterations=1
    )
    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 25)))
    v_lines = cv2.morphologyEx(
        cv2.bitwise_not(clean), cv2.MORPH_OPEN, v_kernel, iterations=1
    )

    grid_lines = cv2.add(h_lines, v_lines)
    # Dilate slightly so subtraction fully covers the lines
    grid_lines = cv2.dilate(grid_lines,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                            iterations=1)

    # Subtract grid lines from the inverted image
    inv_clean = cv2.bitwise_not(clean)
    inv_clean = cv2.subtract(inv_clean, grid_lines)
    clean = cv2.bitwise_not(inv_clean)

    return clean


# ---------------------------------------------------------------------------
# 4. WAVE ISOLATION (Binary → Clean → Skeleton)
# ---------------------------------------------------------------------------
def isolate_wave(grid_removed: np.ndarray) -> tuple:
    """
    Returns (binary_mask, cleaned_mask, skeleton).
    """
    # 4a. Adaptive Gaussian threshold
    binary = cv2.adaptiveThreshold(
        grid_removed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=51, C=12,
    )

    # 4b. Morphological cleaning
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Opening to remove tiny speckles
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_sm, iterations=1)

    # Closing to fill small gaps in the wave
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_sm, iterations=2)

    # 4c. Connected-component area filtering – keep only large components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        # Keep components larger than 1% of the biggest one
        max_area = areas.max()
        threshold_area = max(max_area * 0.01, 100)
        keep_mask = np.zeros_like(cleaned)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= threshold_area:
                keep_mask[labels == i] = 255
        cleaned = keep_mask

    # 4d. Skeletonization (Zhang-Suen thinning)
    try:
        skeleton = cv2.ximgproc.thinning(
            cleaned, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    except AttributeError:
        # Fallback morphological skeletonization
        skeleton = np.zeros_like(cleaned)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = cleaned.copy()
        while True:
            eroded = cv2.erode(img, elem)
            temp = cv2.dilate(eroded, elem)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break

    return binary, cleaned, skeleton


# ---------------------------------------------------------------------------
# 5. WAVEFORM DIGITIZATION
# ---------------------------------------------------------------------------
def detect_grid_spacing(enhanced_gray: np.ndarray) -> float:
    """
    Attempt to detect the small-square grid spacing in pixels
    by finding the dominant frequency of horizontal line spacing.
    Falls back to 20px if detection fails.
    """
    # Sum along columns → 1-D profile
    profile = np.mean(enhanced_gray, axis=1).astype(np.float64)
    profile -= np.mean(profile)
    # Auto-correlation to find periodicity
    corr = np.correlate(profile, profile, mode='full')
    corr = corr[len(corr) // 2:]  # positive lags

    # Find first peak after lag 5
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(corr, distance=5)
    if len(peaks) > 0:
        return float(peaks[0])
    return 20.0  # default fallback


def digitize_waveform(skeleton: np.ndarray, grid_spacing_px: float) -> dict:
    """
    Convert skeleton pixel coordinates → calibrated signal.
    Standard ECG: 25 mm/s, 10 mm/mV
    Small square = 1 mm = 0.04 s (time) = 0.1 mV (voltage)
    """
    y_coords, x_coords = np.where(skeleton > 0)
    if len(x_coords) == 0:
        return {"x_px": [], "y_px": [], "time_s": [], "voltage_mv": [],
                "sample_rate_hz": 0, "grid_spacing_px": grid_spacing_px}

    # Average y per x column
    coord_dict = {}
    for x, y in zip(x_coords, y_coords):
        coord_dict.setdefault(x, []).append(y)

    sorted_x = sorted(coord_dict.keys())
    x_px = np.array(sorted_x, dtype=np.float64)
    y_px = np.array([np.mean(coord_dict[x]) for x in sorted_x], dtype=np.float64)

    # Baseline = median y (the isoelectric line is the most common y)
    baseline_y = float(np.median(y_px))

    # Calibration: 1 small square = grid_spacing_px pixels
    mm_per_px = 1.0 / grid_spacing_px  # 1 mm per small square
    time_per_px = 0.04 * mm_per_px     # seconds  (25 mm/s → 0.04 s/mm)
    mv_per_px = 0.1 * mm_per_px        # millivolts (10 mm/mV → 0.1 mV/mm)

    time_s = (x_px - x_px[0]) * time_per_px
    voltage_mv = (baseline_y - y_px) * mv_per_px  # inverted y axis

    # Estimate sample rate
    if len(time_s) > 1:
        dt = np.median(np.diff(time_s))
        sample_rate = 1.0 / dt if dt > 0 else 250.0
    else:
        sample_rate = 250.0

    return {
        "x_px": x_px.tolist(),
        "y_px": y_px.tolist(),
        "time_s": time_s.tolist(),
        "voltage_mv": voltage_mv.tolist(),
        "baseline_y": baseline_y,
        "sample_rate_hz": round(float(sample_rate), 2),
        "grid_spacing_px": round(grid_spacing_px, 2),
    }


# ---------------------------------------------------------------------------
# PUBLIC: Full pipeline orchestrator
# ---------------------------------------------------------------------------
def run_full_pipeline(image: np.ndarray) -> dict:
    """
    Executes the complete 5-stage image processing pipeline.
    Returns a dict with all intermediate images and digitised data.
    """
    # Stage 1 – Quality
    quality = assess_quality(image)

    # Stage 2 – Pre-processing
    colour, gray, denoised, enhanced = preprocess(image)

    # Stage 3 – Grid Removal
    grid_removed = remove_grid(colour, enhanced)

    # Stage 4 – Wave Isolation
    binary, cleaned, skeleton = isolate_wave(grid_removed)

    # Stage 5 – Digitization
    grid_spacing = detect_grid_spacing(enhanced)
    signal_data = digitize_waveform(skeleton, grid_spacing)

    # Build annotated overlay on original colour image
    overlay = colour.copy()
    skel_pts = np.column_stack(np.where(skeleton > 0))  # (y, x)
    for (y, x) in skel_pts:
        cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

    return {
        "quality": quality,
        "images": {
            "original": colour,
            "preprocessed": cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
            "grid_removed": cv2.cvtColor(grid_removed, cv2.COLOR_GRAY2BGR),
            "binary": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
            "cleaned": cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR),
            "skeleton": cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR),
            "overlay": overlay,
        },
        "signal": signal_data,
    }
