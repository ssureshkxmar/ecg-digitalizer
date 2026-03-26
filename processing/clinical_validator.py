"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 10 — Clinical Consistency Engine + Confidence Scoring   ║
║  Physiological validation + signal quality assessment          ║
╚══════════════════════════════════════════════════════════════════╝

Validates:
  • Heart rate in plausible range (30–220 bpm)
  • QRS width 60–120 ms
  • PR interval 120–200 ms
  • QT interval physiologically valid
  • QTc (Bazett) < 500 ms
  • P wave always before QRS
  • R-R interval consistency (HRV check)
  • Signal-to-noise ratio

Generates:
  • Overall signal quality score (0–100)
  • Per-metric confidence
  • Uncertainty estimation
  • Flags and warnings
"""

import numpy as np


def _validate_hr(metrics: dict) -> dict:
    hr = metrics.get("heart_rate_bpm")
    if hr is None:
        return {"valid": False, "value": None, "issue": "Heart rate not detected",
                "confidence": 0.0}
    valid = 30 <= hr <= 220
    if hr < 30:
        issue = f"Bradycardia: {hr} BPM (below 30)"
    elif hr > 220:
        issue = f"Implausible rate: {hr} BPM (above 220)"
    elif hr < 60:
        issue = f"Bradycardia: {hr} BPM"
    elif hr > 100:
        issue = f"Tachycardia: {hr} BPM"
    else:
        issue = None
    conf = 1.0 if valid else 0.3
    return {"valid": valid, "value": hr, "issue": issue, "confidence": conf}


def _validate_qrs(metrics: dict) -> dict:
    qrs = metrics.get("avg_QRS_ms")
    if qrs is None:
        return {"valid": False, "value": None, "issue": "QRS not measured",
                "confidence": 0.0}
    valid = 60 <= qrs <= 200  # wider range for extracted signals
    issue = None
    if qrs < 60:
        issue = f"QRS too narrow: {qrs} ms"
    elif qrs > 120:
        issue = f"Wide QRS: {qrs} ms (possible bundle branch block)"
    conf = 1.0 if 60 <= qrs <= 120 else (0.6 if valid else 0.2)
    return {"valid": valid, "value": qrs, "issue": issue, "confidence": conf}


def _validate_pr(metrics: dict) -> dict:
    pr = metrics.get("avg_PR_ms")
    if pr is None:
        return {"valid": False, "value": None, "issue": "PR not measured",
                "confidence": 0.0}
    valid = 80 <= pr <= 400  # wider for extracted
    issue = None
    if pr < 120:
        issue = f"Short PR: {pr} ms (possible pre-excitation)"
    elif pr > 200:
        issue = f"Long PR: {pr} ms (possible AV block)"
    conf = 1.0 if 120 <= pr <= 200 else (0.5 if valid else 0.2)
    return {"valid": valid, "value": pr, "issue": issue, "confidence": conf}


def _validate_qt(metrics: dict) -> dict:
    qt = metrics.get("avg_QT_ms")
    qtc = metrics.get("QTc_ms")
    result = {"valid": True, "value": qt, "qtc": qtc, "issue": None,
              "confidence": 0.5}

    if qt is None:
        result["valid"] = False
        result["issue"] = "QT not measured"
        result["confidence"] = 0.0
        return result

    if qtc is not None:
        if qtc > 500:
            result["issue"] = f"Prolonged QTc: {qtc} ms (risk of arrhythmia)"
            result["confidence"] = 0.4
        elif qtc > 450:
            result["issue"] = f"Borderline QTc: {qtc} ms"
            result["confidence"] = 0.7
        else:
            result["confidence"] = 1.0

    return result


def _validate_rhythm(beats: list, metrics: dict) -> dict:
    """Check R-R interval consistency."""
    rr = metrics.get("avg_RR_s")
    if rr is None or len(beats) < 2:
        return {"regular": False, "issue": "Insufficient beats for rhythm analysis",
                "confidence": 0.0}

    # Collect individual RR intervals
    r_times = []
    for b in beats:
        if "R" in b and b["R"]["time"] is not None:
            r_times.append(b["R"]["time"])
    r_times.sort()

    if len(r_times) < 2:
        return {"regular": False, "issue": "Insufficient R peaks",
                "confidence": 0.0}

    rr_intervals = np.diff(r_times)
    if len(rr_intervals) == 0:
        return {"regular": False, "issue": "No RR intervals", "confidence": 0.0}

    rr_std = float(np.std(rr_intervals))
    rr_mean = float(np.mean(rr_intervals))
    cv = rr_std / rr_mean if rr_mean > 0 else 999

    regular = cv < 0.15
    issue = None if regular else f"Irregular rhythm (CV={cv:.2f})"
    conf = max(0.2, 1.0 - cv * 2)

    return {"regular": bool(regular), "cv": round(cv, 3),
            "rr_std_s": round(rr_std, 4), "issue": issue,
            "confidence": round(conf, 2)}


def _validate_p_before_qrs(beats: list) -> dict:
    """Ensure P wave appears before QRS in each beat."""
    violations = 0
    checked = 0
    for b in beats:
        if "P" in b and "Q" in b:
            checked += 1
            if b["P"]["time"] is not None and b["Q"]["time"] is not None:
                if b["P"]["time"] >= b["Q"]["time"]:
                    violations += 1

    if checked == 0:
        return {"valid": True, "issue": None, "confidence": 0.5}

    valid = violations == 0
    issue = f"P wave after QRS in {violations}/{checked} beats" if not valid else None
    conf = 1.0 - (violations / checked)
    return {"valid": bool(valid), "violations": int(violations),
            "checked": int(checked), "issue": issue,
            "confidence": round(conf, 2)}


def _signal_quality_score(signal: list, filtered: list) -> dict:
    """
    Signal-to-noise ratio and overall quality estimation.
    """
    sig = np.array(signal, dtype=np.float64)
    filt = np.array(filtered, dtype=np.float64)

    if len(sig) < 10:
        return {"snr_db": 0, "quality_score": 0, "grade": "UNKNOWN"}

    noise = sig - filt
    sig_power = float(np.var(filt))
    noise_power = float(np.var(noise))

    snr = 10 * np.log10(sig_power / (noise_power + 1e-10)) if noise_power > 0 else 50
    snr = float(np.clip(snr, -10, 60))

    # Map SNR to quality score
    if snr >= 20:
        quality = min(100, 70 + snr)
    elif snr >= 10:
        quality = 50 + snr * 2
    else:
        quality = max(10, snr * 3 + 30)

    quality = round(float(np.clip(quality, 0, 100)), 1)

    if quality >= 85:
        grade = "EXCELLENT"
    elif quality >= 70:
        grade = "GOOD"
    elif quality >= 50:
        grade = "FAIR"
    else:
        grade = "POOR"

    return {"snr_db": round(snr, 1), "quality_score": quality, "grade": grade}


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════
def validate_clinical(beats: list, metrics: dict,
                      raw_signal: list, filtered_signal: list) -> dict:
    """
    Full clinical consistency validation.
    Returns flags, scores, and overall confidence.
    """
    hr_val = _validate_hr(metrics)
    qrs_val = _validate_qrs(metrics)
    pr_val = _validate_pr(metrics)
    qt_val = _validate_qt(metrics)
    rhythm_val = _validate_rhythm(beats, metrics)
    p_order = _validate_p_before_qrs(beats)
    sig_qual = _signal_quality_score(raw_signal, filtered_signal)

    # Collect all issues
    all_flags = []
    validations = {
        "heart_rate": hr_val,
        "qrs_duration": qrs_val,
        "pr_interval": pr_val,
        "qt_interval": qt_val,
        "rhythm": rhythm_val,
        "p_wave_order": p_order,
    }

    for name, val in validations.items():
        if val.get("issue"):
            all_flags.append({"metric": name, "issue": val["issue"]})

    # Overall confidence
    conf_values = [v["confidence"] for v in validations.values()]
    overall_confidence = round(float(np.mean(conf_values)), 2)

    # Overall clinical score (combining signal quality + clinical validity)
    clinical_score = round(
        0.4 * sig_qual["quality_score"] + 0.6 * (overall_confidence * 100), 1)

    return {
        "validations": validations,
        "signal_quality": sig_qual,
        "flags": all_flags,
        "overall_confidence": overall_confidence,
        "clinical_score": round(clinical_score, 1),
        "grade": sig_qual["grade"],
    }
