"""
╔══════════════════════════════════════════════════════════════════════════╗
║         DIGITALIZER — MASTER PIPELINE ORCHESTRATOR                     ║
║         10-Stage Medical-Grade ECG Digitization Architecture           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Stage  1: Intelligent Quality Assessment AI                          ║
║  Stage  2: Super-Resolution Enhancement                               ║
║  Stage  3: Advanced Multi-Strategy Grid Suppression                    ║
║  Stage  4: Subpixel Wave Extraction                                    ║
║  Stage  5: Multi-Model Fusion                                          ║
║  Stage  6: Confidence-Weighted Signal Consensus                        ║
║  Stage  7: Adaptive Calibration Engine                                 ║
║  Stage  8: Biomedical Signal Optimization                              ║
║  Stage  9: Advanced Clinical-Grade Wave Detection (NeuroKit2)          ║
║  Stage 10: Clinical Consistency Engine                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np

from processing.quality_ai import assess_image_quality
from processing.enhancement import enhance_image
from processing.grid_suppressor import suppress_grid
from processing.multi_extraction import extract_waveform
from processing.calibration import calibrate_signal
from processing.signal_processor import optimize_signal
from processing.wave_detector import detect_waves
from processing.clinical_validator import validate_clinical


def _remove_borders(image: np.ndarray) -> np.ndarray:
    """
    Detect and remove thick black borders / separator lines from ECG images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    dark = (gray < 60).astype(np.uint8) * 255
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 3, 100), 1))
    h_borders = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 3, 100)))
    v_borders = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)

    borders = cv2.add(h_borders, v_borders)
    borders = cv2.dilate(borders,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                         iterations=2)

    if cv2.countNonZero(borders) > 0:
        result = cv2.inpaint(image, borders, 7, cv2.INPAINT_TELEA)
        return result
    return image


def _select_best_lead(regions: list, all_region_data: list) -> int:
    """
    Select the best single lead for clinical analysis.
    
    Strategy: Pick the lead row that is most likely the "rhythm strip" 
    (typically Lead II at the bottom of a 12-lead ECG, or the longest/widest row).
    
    Selection criteria:
    1. Prefer the longest (widest) lead — rhythm strips span full width
    2. Prefer leads with more signal points
    3. Prefer leads closer to the bottom (Lead II rhythm strip is usually last)
    """
    if len(all_region_data) <= 1:
        return 0
    
    scores = []
    for i, (region, data) in enumerate(zip(regions, all_region_data)):
        y1, y2 = region
        height = y2 - y1
        n_points = len(data.get("x", []))
        
        if n_points == 0:
            scores.append(-1)
            continue
        
        x_arr = data["x"]
        x_span = float(np.max(x_arr) - np.min(x_arr)) if len(x_arr) > 1 else 0
        
        # Score: width of the trace + bonus for being bottom row + point count bonus
        position_bonus = i * 0.1  # slight preference for lower rows (rhythm strip)
        width_score = x_span / 1000.0  # normalize
        point_score = n_points / 10000.0  # normalize
        
        score = width_score + position_bonus + point_score
        scores.append(score)
    
    best_idx = int(np.argmax(scores))
    return best_idx


def run_pipeline(image: np.ndarray, config: dict = None) -> dict:
    """
    Execute the full 10-stage pipeline with optional configuration.
    
    KEY FIX: For clinical analysis (HR, wave detection), we use only
    the best single lead (typically the rhythm strip) to avoid 
    multi-lead concatenation artifacts that cause incorrect HR.
    """
    if config is None:
        config = {}

    # ─── PRE-STEP: Border/separator removal ──────────────────────────
    image_clean = _remove_borders(image)

    # ─── STAGE 1: Quality Assessment ─────────────────────────────────
    quality = assess_image_quality(image)

    # ─── STAGE 2: Enhancement ────────────────────────────────────────
    enhanced = enhance_image(image_clean)
    colour = enhanced["enhanced_colour"]
    gray = enhanced["enhanced_gray"]

    # ─── STAGE 3: Grid Suppression ───────────────────────────────────
    if config.get("grid_suppression", True):
        grid_result = suppress_grid(colour, gray)
        grid_removed = grid_result["fused"]
        grid_strategies = grid_result["strategies"]
    else:
        grid_removed = gray
        grid_strategies = {"colour": gray.copy(), "fft": gray.copy(), "morphological": gray.copy()}

    # ─── STAGES 4-6: Multi-Model Extraction + Fusion ────────────────
    extraction = extract_waveform(grid_removed)
    fused = extraction["fused"]
    x_px = fused["x"]
    y_px = fused["y_smooth"]
    confidence = fused["confidence"]
    orig_x = fused.get("original_x", x_px)
    orig_y = fused.get("original_y", y_px)

    # ─── BEST LEAD SELECTION ─────────────────────────────────────────
    # For clinical analysis, use only the best single lead to avoid
    # multi-lead concatenation artifacts
    lead_regions = extraction.get("lead_regions", [])
    lead_data_list = extraction.get("lead_data_list", [])
    
    if lead_data_list and len(lead_data_list) > 1:
        best_lead = _select_best_lead(lead_regions, lead_data_list)
        best_data = lead_data_list[best_lead]
        # Use only the best lead for calibration and analysis
        analysis_x = best_data["x"]
        analysis_y = best_data["y_smooth"]
    else:
        # Single lead or no lead separation — use all data
        analysis_x = x_px
        analysis_y = y_px

    # ─── STAGE 7: Calibration (on best lead only) ────────────────────
    cal = calibrate_signal(analysis_x, analysis_y, gray)

    # ─── STAGE 8: Signal Processing ──────────────────────────────────
    processed = optimize_signal(cal["voltage_mv"], cal["sample_rate_hz"], config)

    # ─── STAGE 9: Wave Detection (NeuroKit2 powered) ─────────────────
    waves = detect_waves(processed["filtered"], cal["time_s"],
                         processed["sample_rate"])

    # ─── STAGE 10: Clinical Validation ───────────────────────────────
    clinical = validate_clinical(
        waves["beats"], waves["metrics"],
        processed["raw"], processed["filtered"]
    )

    # ─── Build output images ─────────────────────────────────────────
    overlay = colour.copy()
    skeleton_mask = extraction["skeleton"]
    
    if skeleton_mask is not None and np.any(skeleton_mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thick_skeleton = cv2.dilate(skeleton_mask, kernel, iterations=1)
        overlay[thick_skeleton > 0] = (0, 255, 0)

    # Confidence heatmap
    conf_map = colour.copy()
    for i in range(len(orig_x)):
        c = float(confidence[i]) if i < len(confidence) else 0.5
        r = int((1 - c) * 255)
        g = int(c * 255)
        cv2.circle(conf_map, (int(orig_x[i]), int(orig_y[i])), 2, (0, g, r), -1)

    images = {
        "original": image,
        "enhanced": colour,
        "grid_removed_fused": cv2.cvtColor(grid_removed, cv2.COLOR_GRAY2BGR),
        "grid_colour": cv2.cvtColor(grid_strategies["colour"], cv2.COLOR_GRAY2BGR),
        "grid_fft": cv2.cvtColor(grid_strategies["fft"], cv2.COLOR_GRAY2BGR),
        "grid_morphological": cv2.cvtColor(grid_strategies["morphological"], cv2.COLOR_GRAY2BGR),
        "binary": cv2.cvtColor(extraction["binary"], cv2.COLOR_GRAY2BGR),
        "cleaned": cv2.cvtColor(extraction["cleaned"], cv2.COLOR_GRAY2BGR),
        "skeleton": cv2.cvtColor(extraction["skeleton"], cv2.COLOR_GRAY2BGR),
        "overlay": overlay,
        "confidence_map": conf_map,
    }

    return {
        "quality": quality,
        "images": images,
        "signal": {
            "x_px": x_px.tolist() if isinstance(x_px, np.ndarray) else [],
            "original_x": orig_x.tolist() if isinstance(orig_x, np.ndarray) else [],
            "original_y": orig_y.tolist() if isinstance(orig_y, np.ndarray) else [],
            "y_px": y_px.tolist() if isinstance(y_px, np.ndarray) else [],
            "time_s": cal["time_s"],
            "voltage_mv": cal["voltage_mv"],
            "sample_rate_hz": cal["sample_rate_hz"],
            "baseline_y": cal["baseline_y"],
        },
        "processed_signal": processed,
        "wave_detection": waves,
        "clinical": clinical,
        "calibration": cal.get("calibration", {}),
        "grid_info": cal.get("grid", {}),
        "extraction_info": {
            "method_results": extraction["method_results"],
            "method_agreement": fused["method_agreement"],
            "total_points": len(x_px),
            "best_lead": best_lead if lead_data_list and len(lead_data_list) > 1 else 0,
            "num_leads": len(lead_regions) if lead_regions else 1,
        },
    }
