"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGES 4–6 — Subpixel Wave Extraction + Multi-Model Fusion   ║
║  Three independent extraction methods fused by confidence      ║
╚══════════════════════════════════════════════════════════════════╝

CRITICAL: This module includes intelligent filtering to reject:
  • Image borders and margins
  • Text labels and annotations
  • Grid remnants and artifacts
  • Compact blobs that are not waveforms

Only long, horizontally-oriented traces survive the filtering.
"""

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import heapq


# ═══════════════════════════════════════════════════════════════════
# SMART BINARISATION + BORDER/NOISE REJECTION
# ═══════════════════════════════════════════════════════════════════
def _smart_binarise(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive binarisation with automatic border and margin removal.
    """
    h, w = gray.shape

    # --- Step 1: Adaptive threshold ---
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=10)

    # --- Step 2: Remove image borders (thick lines at edges) ---
    # Zero out a margin region at edges where borders/text usually live
    margin = max(int(min(h, w) * 0.02), 5)
    binary[:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, -margin:] = 0

    # --- Step 3: Morphological cleaning ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Opening removes tiny dots
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing fills small gaps in the trace
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary


def _filter_components(binary: np.ndarray) -> np.ndarray:
    """
    Intelligent connected-component filtering.
    Keep only components that look like ECG traces:
      - Eliminate components touching all 4 borders (full-image frames)
      - Eliminate very compact shapes (text characters, dots)
      - Prefer horizontally elongated components (ECG traces are wide)
      - Eliminate very small components (noise)
    """
    h, w = binary.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    if num_labels <= 1:
        return binary

    # Collect component metrics
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return binary

    total_area = h * w
    max_area = areas.max()
    result = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        # --- Reject criteria ---

        # Too small: noise/dots (less than 0.05% of image)
        if area < total_area * 0.0005:
            continue

        # Too large: probably the entire grid or background
        if area > total_area * 0.5:
            continue

        # Component spans nearly the full border? (frame/border)
        touches_left = cx <= 3
        touches_right = cx + cw >= w - 3
        touches_top = cy <= 3
        touches_bottom = cy + ch >= h - 3
        if (touches_left and touches_right and touches_top) or \
           (touches_left and touches_right and touches_bottom) or \
           (touches_top and touches_bottom and touches_left) or \
           (touches_top and touches_bottom and touches_right):
            continue

        # Compactness check: reject compact blobs (text chars)
        # ECG traces have low compactness (area relative to bounding box)
        bbox_area = cw * ch
        if bbox_area > 0:
            fill_ratio = area / bbox_area
            aspect_ratio = cw / (ch + 1e-6)

            # Very compact and nearly square = text character
            if fill_ratio > 0.6 and aspect_ratio < 3 and area < total_area * 0.01:
                continue

            # Very tall narrow components = vertical lines/borders
            if aspect_ratio < 0.15 and ch > h * 0.3:
                continue

        # Minimum width: ECG traces span a good portion of the image
        # But allow smaller traces too (individual leads)
        min_width = w * 0.05
        if cw < min_width and area < max_area * 0.05:
            continue

        result[labels == i] = 255

    return result


# ═══════════════════════════════════════════════════════════════════
# METHOD 1: Classical Skeleton Tracing
# ═══════════════════════════════════════════════════════════════════
def _skeleton_trace(cleaned: np.ndarray) -> dict:
    """Zhang-Suen thinning → 1-pixel trace → coordinates."""
    if cv2.countNonZero(cleaned) == 0:
        return {"x": np.array([]), "y": np.array([]), "confidence": np.array([]), "skeleton": np.zeros_like(cleaned)}

    try:
        skeleton = cv2.ximgproc.thinning(
            cleaned, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
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

    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return {"x": np.array([]), "y": np.array([]), "confidence": np.array([]),
                "skeleton": skeleton}

    coord_dict = {}
    for x, y in zip(xs, ys):
        coord_dict.setdefault(int(x), []).append(int(y))

    sorted_x = sorted(coord_dict.keys())
    x_arr = np.array(sorted_x, dtype=np.float64)
    y_arr = np.array([np.mean(coord_dict[x]) for x in sorted_x], dtype=np.float64)
    conf = np.array([1.0 / (1.0 + np.std(coord_dict[x]))
                     for x in sorted_x], dtype=np.float64)

    return {"x": x_arr, "y": y_arr, "confidence": conf, "skeleton": skeleton}


# ═══════════════════════════════════════════════════════════════════
# METHOD 2: Column Intensity Peak Tracing (Gaussian Subpixel)
# ═══════════════════════════════════════════════════════════════════
def _column_peak_trace(cleaned: np.ndarray) -> dict:
    """For each x column, find peak y using Gaussian subpixel fitting."""
    h, w = cleaned.shape
    x_list, y_list, conf_list = [], [], []

    for x in range(w):
        col = cleaned[:, x].astype(np.float64)
        if np.max(col) < 10:
            continue

        peak_idx = int(np.argmax(col))
        peak_val = col[peak_idx]

        # Gaussian subpixel refinement
        if 1 <= peak_idx < len(col) - 1 and peak_val > 20:
            y_m1 = float(col[peak_idx - 1])
            y_0 = float(col[peak_idx])
            y_p1 = float(col[peak_idx + 1])
            denom = 2.0 * (2 * y_0 - y_m1 - y_p1)
            if abs(denom) > 1e-6:
                offset = (y_m1 - y_p1) / denom
                sub_y = peak_idx + np.clip(offset, -0.5, 0.5)
            else:
                sub_y = float(peak_idx)
        else:
            sub_y = float(peak_idx)

        conf = min(1.0, peak_val / 255.0)
        x_list.append(float(x))
        y_list.append(sub_y)
        conf_list.append(conf)

    return {"x": np.array(x_list), "y": np.array(y_list),
            "confidence": np.array(conf_list)}


# ═══════════════════════════════════════════════════════════════════
# METHOD 3: Graph-Based Shortest Path
# ═══════════════════════════════════════════════════════════════════
def _graph_shortest_path(cleaned: np.ndarray) -> dict:
    """Dijkstra shortest path from left to right through high-intensity pixels."""
    h, w = cleaned.shape
    col_sums = np.sum(cleaned, axis=0)
    active_cols = np.where(col_sums > 0)[0]
    if len(active_cols) < 10:
        return {"x": np.array([]), "y": np.array([]), "confidence": np.array([])}

    start_x = int(active_cols[0])
    end_x = int(active_cols[-1])

    cost = 255.0 - cleaned.astype(np.float64) + 1.0
    start_y = int(np.argmax(cleaned[:, start_x]))

    dist = np.full((h, w), np.inf, dtype=np.float64)
    visited = np.zeros((h, w), dtype=bool)
    parent = np.full((h, w, 2), -1, dtype=np.int32)
    dist[start_y, start_x] = 0
    pq = [(0.0, start_y, start_x)]

    dy_dx = [(-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)]
    move_costs = [1.5, 1.5, 1.0, 1.2, 1.2]
    band = min(100, h // 3)

    while pq:
        d, y, x = heapq.heappop(pq)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if x >= end_x:
            break
        for (dy, dx), mc in zip(dy_dx, move_costs):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if abs(ny - start_y) > band:
                    continue
                nd = d + cost[ny, nx] * mc
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    parent[ny, nx] = [y, x]
                    heapq.heappush(pq, (nd, ny, nx))

    end_col_dist = dist[:, end_x]
    end_y = int(np.argmin(end_col_dist))
    if end_col_dist[end_y] == np.inf:
        return {"x": np.array([]), "y": np.array([]), "confidence": np.array([])}

    path = []
    cy, cx = end_y, end_x
    while cx != -1 and cy != -1:
        path.append((cx, cy))
        py, px = parent[cy, cx]
        cy, cx = int(py), int(px)
    path.reverse()

    if len(path) < 10:
        return {"x": np.array([]), "y": np.array([]), "confidence": np.array([])}

    coord_dict = {}
    for x, y in path:
        coord_dict.setdefault(x, []).append(y)

    sorted_x = sorted(coord_dict.keys())
    x_arr = np.array(sorted_x, dtype=np.float64)
    y_arr = np.array([np.mean(coord_dict[x]) for x in sorted_x], dtype=np.float64)
    conf_arr = np.ones_like(x_arr) * 0.8

    return {"x": x_arr, "y": y_arr, "confidence": conf_arr}


# ═══════════════════════════════════════════════════════════════════
# B-SPLINE SMOOTHING
# ═══════════════════════════════════════════════════════════════════
def _bspline_smooth(x, y, s_factor=0.5):
    if len(x) < 10:
        return y.copy()
    _, unique_idx = np.unique(x, return_index=True)
    x_u, y_u = x[unique_idx], y[unique_idx]
    if len(x_u) < 10:
        return y.copy()
    try:
        spl = UnivariateSpline(x_u, y_u, s=s_factor * len(x_u),
                               k=min(3, len(x_u) - 1))
        return spl(x)
    except Exception:
        return y.copy()


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE-WEIGHTED FUSION
# ═══════════════════════════════════════════════════════════════════
def _fuse_methods(methods):
    all_x = set()
    for m in methods:
        if len(m["x"]) > 0:
            all_x.update(m["x"].astype(int).tolist())
    if not all_x:
        return {"x": np.array([]), "y": np.array([]),
                "confidence": np.array([]), "method_agreement": 0.0}

    sorted_x = sorted(all_x)
    fused_x, fused_y, fused_conf = [], [], []

    for x in sorted_x:
        ys, ws = [], []
        for m in methods:
            if len(m["x"]) == 0:
                continue
            idx = np.where(m["x"].astype(int) == x)[0]
            if len(idx) > 0:
                ys.append(m["y"][idx[0]])
                ws.append(m["confidence"][idx[0]])
        if not ys:
            continue
        ys, ws = np.array(ys), np.array(ws)
        if len(ys) >= 3:
            med = np.median(ys)
            mad = np.median(np.abs(ys - med))
            if mad > 0:
                keep = np.abs(ys - med) <= 2.5 * mad
                ys, ws = ys[keep], ws[keep]
        if len(ys) == 0:
            continue
        total_w = np.sum(ws) + 1e-10
        fused_x.append(float(x))
        fused_y.append(float(np.sum(ys * ws) / total_w))
        fused_conf.append(float(min(1.0, total_w / len(methods))))

    agreement = sum(1 for x in sorted_x
                    if sum(1 for m in methods if len(m["x"]) > 0 and x in m["x"].astype(int))
                    >= 2) / max(len(sorted_x), 1)

    return {"x": np.array(fused_x), "y": np.array(fused_y),
            "confidence": np.array(fused_conf),
            "method_agreement": round(float(agreement), 3)}


# ═══════════════════════════════════════════════════════════════════
# LEAD SEPARATION — extract individual ECG lead strips
# ═══════════════════════════════════════════════════════════════════
def _detect_lead_rows(cleaned: np.ndarray) -> list:
    """
    Detect horizontal rows of ECG leads.
    Standard 12-lead ECGs have 3-4 horizontal rows of leads.
    Returns list of (y_start, y_end) tuples.
    """
    h, w = cleaned.shape

    # Create horizontal density profile
    row_density = np.sum(cleaned > 0, axis=1).astype(np.float64)

    # Smooth the profile
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(row_density, size=max(h // 40, 5))

    # Find regions where signal exists
    threshold = np.max(smoothed) * 0.05
    active = smoothed > threshold

    # Find contiguous active regions
    regions = []
    in_region = False
    start = 0
    for y in range(h):
        if active[y] and not in_region:
            start = y
            in_region = True
        elif not active[y] and in_region:
            if y - start > h * 0.05:  # region must be at least 5% of height
                regions.append((start, y))
            in_region = False
    if in_region:
        regions.append((start, h))

    if not regions:
        regions = [(0, h)]

    return regions


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════
def extract_waveform(grid_removed_gray: np.ndarray) -> dict:
    """
    Stages 4-6: Smart binarisation + component filtering +
    multi-model extraction + subpixel fusion.
    Handles multi-lead ECGs by extracting the rhythm strip for 1D analysis,
    while providing the full 2D skeleton for perfect visual representation.
    """
    # Step 1: Smart binarisation with border removal
    binary = _smart_binarise(grid_removed_gray)

    # Step 2: Intelligent component filtering
    cleaned = _filter_components(binary)

    # NEW: Get the full-image skeleton for perfect trace visualization
    try:
        full_skeleton = cv2.ximgproc.thinning(
            cleaned, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        full_skeleton = cleaned.copy() # fallback

    # Step 3 & 4: Sequence all regions into a single continuous temporal signal
    regions = _detect_lead_rows(cleaned)
    
    all_fused_x = []
    all_fused_y = []
    all_fused_conf = []
    all_orig_x = []
    all_orig_y = []
    
    # Store per-lead data for best lead selection
    lead_data_list = []
    
    current_x_offset = 0
    total_agreement = 0
    total_elements = 0
    
    method_results = {"skeleton": {"pts": 0, "conf": []}, 
                      "column_peak": {"pts": 0, "conf": []}, 
                      "graph_path": {"pts": 0, "conf": []}}

    for r in regions:
        y1, y2 = r
        row_cleaned = cleaned[y1:y2, :]
        
        # Run extraction ON THIS ROW ONLY
        m1 = _skeleton_trace(row_cleaned)
        m2 = _column_peak_trace(row_cleaned)
        m3 = _graph_shortest_path(row_cleaned)

        # Adjust y-coordinates back to global image space
        for m in [m1, m2, m3]:
            if len(m.get("y", [])) > 0:
                m["y"] = m["y"] + y1

        methods = [m1, m2, m3]
        fused = _fuse_methods(methods)

        if len(fused["x"]) >= 10:
            fused["y_smooth"] = _bspline_smooth(fused["x"], fused["y"], s_factor=0.3)
        else:
            fused["y_smooth"] = fused["y"].copy()
        
        # Store per-lead data for best lead selection
        lead_data_list.append({
            "x": fused["x"].copy(),
            "y_smooth": fused["y_smooth"].copy(),
            "confidence": fused["confidence"].copy(),
            "method_agreement": fused.get("method_agreement", 0),
        })
            
        all_orig_x.extend(fused["x"].tolist())
        all_orig_y.extend(fused["y_smooth"].tolist())
        
        # Shift x for continuous 1D signal
        shifted_x = fused["x"] + current_x_offset
        all_fused_x.extend(shifted_x.tolist())
        all_fused_y.extend(fused["y_smooth"].tolist())
        all_fused_conf.extend(fused["confidence"].tolist())
        
        if len(fused["x"]) > 0:
            current_x_offset = shifted_x[-1] + 1  # Next region starts after this one
            total_agreement += fused.get("method_agreement", 0) * len(fused["x"])
            total_elements += len(fused["x"])
            
        method_names = ["skeleton", "column_peak", "graph_path"]
        for name, m in zip(method_names, methods):
            method_results[name]["pts"] += len(m.get("x", []))
            if len(m.get("confidence", [])) > 0:
                method_results[name]["conf"].extend(m["confidence"].tolist())

    method_agreement = total_agreement / total_elements if total_elements > 0 else 0
    
    final_method_results = {}
    for name, data in method_results.items():
        final_method_results[name] = {
            "num_points": data["pts"],
            "avg_confidence": round(float(np.mean(data["conf"])), 3) if data["conf"] else 0,
        }

    return {
        "fused": {
            "x": np.array(all_fused_x),
            "y_smooth": np.array(all_fused_y),
            "confidence": np.array(all_fused_conf),
            "original_x": np.array(all_orig_x),
            "original_y": np.array(all_orig_y),
            "method_agreement": round(float(method_agreement), 3)
        },
        "binary": binary,
        "cleaned": cleaned,
        "skeleton": full_skeleton,
        "method_results": final_method_results,
        "lead_regions": regions,
        "lead_data_list": lead_data_list,
    }
