"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 1 — Intelligent Image Quality Assessment AI             ║
║  Multi-metric CNN-equivalent quality scoring system            ║
╚══════════════════════════════════════════════════════════════════╝

Evaluates:
  • Blur level (Laplacian variance)
  • Motion distortion (directional gradient analysis)
  • Resolution adequacy (DPI estimation)
  • Skew angle (Hough transform rotation detection)
  • Shadow detection (illumination uniformity)
  • Contrast deficiency (histogram spread analysis)
  • Overall signal quality score (0–100)
"""

import cv2
import numpy as np


def _blur_score(gray: np.ndarray) -> dict:
    """Laplacian variance — higher = sharper."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # Map to 0-100 score
    score = min(100, var / 20.0)  # typical sharp ECG > 2000 variance
    return {"laplacian_var": round(var, 2), "score": round(score, 1),
            "is_sharp": var > 100}


def _motion_score(gray: np.ndarray) -> dict:
    """Detect motion blur by comparing horizontal vs vertical gradients."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h_energy = float(np.mean(np.abs(gx)))
    v_energy = float(np.mean(np.abs(gy)))
    ratio = max(h_energy, v_energy) / (min(h_energy, v_energy) + 1e-6)
    has_motion = ratio > 3.0
    score = max(0, 100 - (ratio - 1) * 15)
    return {"h_energy": round(h_energy, 2), "v_energy": round(v_energy, 2),
            "ratio": round(ratio, 2), "has_motion_blur": bool(has_motion),
            "score": round(max(0, score), 1)}


def _resolution_score(h: int, w: int) -> dict:
    """Estimate DPI and resolution adequacy."""
    # Assume standard 12-lead ECG paper is ~11 x 8.5 inches
    dpi_est = max(h, w) / 11.0
    adequate = dpi_est >= 200
    score = min(100, dpi_est / 6.0) if adequate else min(60, dpi_est / 3.5)
    return {"estimated_dpi": int(round(dpi_est)),
            "is_adequate": bool(adequate),
            "resolution": f"{w}x{h}",
            "score": round(score, 1)}


def _skew_score(gray: np.ndarray) -> dict:
    """Detect rotation angle via Hough line transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(gray.shape) // 4)
    angles = []
    if lines is not None:
        for line in lines[:50]:
            rho, theta = line[0]
            angle_deg = float(np.degrees(theta))
            # Normalize to deviation from 0° or 90°
            dev = min(abs(angle_deg), abs(angle_deg - 90),
                      abs(angle_deg - 180))
            if dev < 30:
                angles.append(dev)
    avg_skew = float(np.median(angles)) if angles else 0.0
    needs_correction = avg_skew > 2.0
    score = max(0, 100 - avg_skew * 10)
    return {"skew_degrees": round(avg_skew, 2),
            "needs_correction": bool(needs_correction),
            "score": round(score, 1)}


def _shadow_score(gray: np.ndarray) -> dict:
    """Evaluate illumination uniformity across the image."""
    h, w = gray.shape
    # Split into quadrants and compare mean intensities
    quads = [
        gray[:h//2, :w//2], gray[:h//2, w//2:],
        gray[h//2:, :w//2], gray[h//2:, w//2:]
    ]
    means = [float(np.mean(q)) for q in quads]
    spread = max(means) - min(means)
    has_shadow = spread > 40
    score = max(0, 100 - spread * 1.2)
    return {"quad_means": [round(m, 1) for m in means],
            "intensity_spread": round(spread, 1),
            "has_shadow": bool(has_shadow),
            "score": round(score, 1)}


def _contrast_score(gray: np.ndarray) -> dict:
    """Histogram spread and dynamic range analysis."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    # Find 5th and 95th percentile
    cumsum = np.cumsum(hist) / hist.sum()
    p5 = int(np.searchsorted(cumsum, 0.05))
    p95 = int(np.searchsorted(cumsum, 0.95))
    dynamic_range = p95 - p5
    std_dev = float(np.std(gray.astype(np.float64)))
    adequate = dynamic_range > 80
    score = min(100, dynamic_range / 2.5)
    return {"dynamic_range": int(dynamic_range),
            "percentiles": [int(p5), int(p95)],
            "std_dev": round(std_dev, 2),
            "is_adequate": bool(adequate),
            "score": round(score, 1)}


def _waveform_visibility(gray: np.ndarray) -> dict:
    """Detect if ECG waveform traces are visible using edge density analysis."""
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(cv2.countNonZero(edges)) / (gray.shape[0] * gray.shape[1])
    # ECG images typically have edge density 0.02–0.15
    visible = 0.01 < edge_density < 0.3
    score = 100.0 if visible else (50.0 if edge_density > 0.005 else 20.0)
    return {"edge_density": round(edge_density, 4),
            "waveform_visible": bool(visible),
            "score": round(score, 1)}


# ─── PUBLIC API ──────────────────────────────────────────────────────────────
def assess_image_quality(image: np.ndarray) -> dict:
    """
    Comprehensive image quality assessment.
    Returns per-metric scores and an overall quality score (0-100).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    blur = _blur_score(gray)
    motion = _motion_score(gray)
    resolution = _resolution_score(h, w)
    skew = _skew_score(gray)
    shadow = _shadow_score(gray)
    contrast = _contrast_score(gray)
    visibility = _waveform_visibility(gray)

    # Weighted overall score
    weights = {
        "blur": 0.20, "motion": 0.10, "resolution": 0.15,
        "skew": 0.10, "shadow": 0.10, "contrast": 0.15,
        "visibility": 0.20,
    }
    scores = {
        "blur": blur["score"], "motion": motion["score"],
        "resolution": resolution["score"], "skew": skew["score"],
        "shadow": shadow["score"], "contrast": contrast["score"],
        "visibility": visibility["score"],
    }
    overall = sum(scores[k] * weights[k] for k in weights)

    # Quality grade
    if overall >= 85:
        grade = "EXCELLENT"
    elif overall >= 70:
        grade = "GOOD"
    elif overall >= 50:
        grade = "FAIR"
    else:
        grade = "POOR"

    return {
        "overall_score": round(overall, 1),
        "grade": grade,
        "metrics": {
            "blur": blur,
            "motion": motion,
            "resolution": resolution,
            "skew": skew,
            "shadow": shadow,
            "contrast": contrast,
            "visibility": visibility,
        },
        "recommendation": _generate_recommendation(scores, grade),
    }


def _generate_recommendation(scores: dict, grade: str) -> str:
    issues = []
    if scores["blur"] < 50:
        issues.append("Image appears blurry — use higher resolution scan")
    if scores["motion"] < 50:
        issues.append("Motion blur detected — stabilize camera")
    if scores["resolution"] < 50:
        issues.append("Resolution too low — scan at ≥300 DPI")
    if scores["skew"] < 70:
        issues.append("Image is skewed — align paper before scanning")
    if scores["shadow"] < 60:
        issues.append("Uneven lighting / shadows detected")
    if scores["contrast"] < 50:
        issues.append("Poor contrast — ensure dark trace on light paper")
    if scores["visibility"] < 50:
        issues.append("Waveform barely visible — check image content")

    if not issues:
        return "Image quality is excellent for ECG extraction."
    return " | ".join(issues)
