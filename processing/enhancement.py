"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 2 — Super-Resolution & Enhancement Engine               ║
║  Multi-pass image enhancement for thin ECG line recovery       ║
╚══════════════════════════════════════════════════════════════════╝

Implements:
  • Intelligent upscaling (2×) with edge preservation
  • Unsharp masking for edge sharpening
  • CLAHE contrast enhancement
  • Detail-preserving denoising (bilateral + non-local means)
  • Perspective / rotation correction
"""

import cv2
import numpy as np


def _correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Detect paper boundaries and apply perspective correction.
    Uses contour detection to find the largest rectangle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = image.shape[0] * image.shape[1]

    # Only correct if contour covers > 30% of image (likely the paper)
    if area < 0.3 * img_area:
        return image

    # Approximate to polygon
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        # Sort points: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(d)]
        ordered[3] = pts[np.argmax(d)]

        w = max(np.linalg.norm(ordered[1] - ordered[0]),
                np.linalg.norm(ordered[2] - ordered[3]))
        h = max(np.linalg.norm(ordered[3] - ordered[0]),
                np.linalg.norm(ordered[2] - ordered[1]))

        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, M, (int(w), int(h)))

    return image


def _correct_rotation(image: np.ndarray) -> np.ndarray:
    """Detect skew angle and deskew the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180,
                           threshold=min(gray.shape) // 3)
    if lines is None:
        return image

    angles = []
    for line in lines[:80]:
        theta = line[0][1]
        angle = float(np.degrees(theta))
        # Focus on near-horizontal or near-vertical lines
        if angle < 10 or angle > 170:
            angles.append(angle if angle < 90 else angle - 180)
        elif 80 < angle < 100:
            angles.append(angle - 90)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return image  # negligible

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_REPLICATE)


def _super_resolve(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Edge-preserving super-resolution via bicubic + Lanczos upscale
    followed by unsharp masking to recover thin ECG lines.
    Only applied when the image is truly small.
    """
    h, w = image.shape[:2]
    # Only upscale truly small images — upscaling larger ones
    # amplifies grid lines and makes extraction worse
    if max(h, w) >= 800:
        return image

    # Lanczos upscale
    up = cv2.resize(image, (w * scale, h * scale),
                    interpolation=cv2.INTER_LANCZOS4)

    # Unsharp mask to sharpen edges
    gaussian = cv2.GaussianBlur(up, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(up, 1.5, gaussian, -0.5, 0)

    return sharpened


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE per channel for colour images, or directly for grayscale."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def _denoise(image: np.ndarray) -> np.ndarray:
    """Multi-pass edge-preserving denoising."""
    # Stage 1: Bilateral filter (edge-preserving)
    d1 = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
    # Stage 2: Non-local means denoising (removes Gaussian noise)
    if len(image.shape) == 3:
        d2 = cv2.fastNlMeansDenoisingColored(d1, None, 6, 6, 7, 21)
    else:
        d2 = cv2.fastNlMeansDenoising(d1, None, 6, 7, 21)
    return d2


# ─── PUBLIC API ──────────────────────────────────────────────────────────────
def enhance_image(image: np.ndarray) -> dict:
    """
    Full Stage 2 enhancement pipeline.
    Returns the enhanced image + all intermediates.
    """
    h, w = image.shape[:2]

    # 2a. Perspective correction
    corrected = _correct_perspective(image)

    # 2b. Rotation correction
    deskewed = _correct_rotation(corrected)

    # 2c. Super-resolution
    upscaled = _super_resolve(deskewed)

    # 2d. Denoising
    denoised = _denoise(upscaled)

    # 2e. Contrast enhancement
    enhanced = _enhance_contrast(denoised)

    # 2f. Generate grayscale for downstream processing
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced

    return {
        "enhanced_colour": enhanced,
        "enhanced_gray": gray,
        "intermediates": {
            "perspective_corrected": corrected,
            "deskewed": deskewed,
            "upscaled": upscaled,
            "denoised": denoised,
        },
    }
