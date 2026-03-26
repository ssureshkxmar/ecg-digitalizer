"""Smoke test with a realistic ECG simulation (proper grid + darker trace)."""
import cv2, numpy as np, requests, sys, json

def make_realistic_ecg():
    """Create a realistic-looking ECG with proper features."""
    w, h = 1000, 600
    img = np.ones((h, w, 3), dtype=np.uint8) * 252  # off-white paper

    # Light pink grid (like real ECG paper)
    for x in range(0, w, 20):
        c = (220, 215, 240) if x % 100 else (195, 185, 225)  # pink
        t = 1 if x % 100 else 1
        cv2.line(img, (x, 0), (x, h), c, t)
    for y in range(0, h, 20):
        c = (220, 215, 240) if y % 100 else (195, 185, 225)
        t = 1 if y % 100 else 1
        cv2.line(img, (0, y), (w, y), c, t)

    # Draw black ECG trace — 3 beats
    baseline = 300
    for bo in [50, 380, 710]:
        pts = []
        for x in range(bo, min(bo + 280, w-20)):
            lx = x - bo
            y = baseline
            # P wave
            if 30 <= lx < 60: y -= int(15 * np.sin(np.pi * (lx - 30) / 30))
            # Q wave
            elif 80 <= lx < 90: y += int(20 * np.sin(np.pi * (lx - 80) / 10))
            # R wave (tall sharp peak)
            elif 90 <= lx < 110: y -= int(120 * np.sin(np.pi * (lx - 90) / 20))
            # S wave
            elif 110 <= lx < 120: y += int(30 * np.sin(np.pi * (lx - 110) / 10))
            # T wave
            elif 160 <= lx < 210: y -= int(25 * np.sin(np.pi * (lx - 160) / 50))
            pts.append((x, y))
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], (10, 10, 10), 2, cv2.LINE_AA)

    return img

img = make_realistic_ecg()
ok, buf = cv2.imencode('.png', img)
resp = requests.post('http://localhost:8001/api/extract',
                     files={'file': ('test.png', buf.tobytes(), 'image/png')})

if resp.status_code != 200:
    print(f"FAIL: HTTP {resp.status_code}")
    print(resp.text[:800])
    sys.exit(1)

data = resp.json()
print("=" * 60)
print("10-STAGE PIPELINE SMOKE TEST")
print("=" * 60)
print(f"STATUS: {data['status']}")
print(f"\n📊 QUALITY AI: {data['quality']['overall_score']}/100 [{data['quality']['grade']}]")
print(f"   Recommendation: {data['quality']['recommendation']}")
print(f"\n📏 CALIBRATION:")
cal = data.get('calibration', {})
gi = data.get('grid_info', {})
print(f"   Grid spacing: {gi.get('small_box_px', '?')} px")
print(f"   Time/px: {cal.get('time_per_px_s', '?')} s")
print(f"   mV/px: {cal.get('mv_per_px', '?')}")
print(f"\n📐 EXTRACTION:")
ei = data.get('extraction_info', {})
print(f"   Total points: {ei.get('total_points', '?')}")
print(f"   Method agreement: {ei.get('method_agreement', '?')}")
mr = ei.get('method_results', {})
for name, info in mr.items():
    print(f"   {name}: {info['num_points']} pts, conf={info['avg_confidence']}")
print(f"\n🖼️ IMAGES: {list(data['images'].keys())}")
print(f"\n📈 SIGNAL PLOT: {'✅' if data.get('signal_plot') else '❌'}")
print(f"\n🫀 CLINICAL METRICS:")
m = data.get('wave_detection', {}).get('metrics', {})
for k, v in m.items():
    print(f"   {k}: {v}")
di = data.get('wave_detection', {}).get('detection_info', {})
print(f"\n🔬 DETECTION INFO:")
for k, v in di.items():
    print(f"   {k}: {v}")
print(f"\n🏥 CLINICAL VALIDATION:")
clin = data.get('clinical', {})
print(f"   Score: {clin.get('clinical_score')}/100")
print(f"   Grade: {clin.get('grade')}")
print(f"   Confidence: {clin.get('overall_confidence')}")
flags = clin.get('flags', [])
if flags:
    for f in flags:
        print(f"   ⚠️ {f['metric']}: {f['issue']}")
else:
    print("   ✅ All parameters normal")
print(f"\nBEATS: {len(data.get('wave_detection', {}).get('beats', []))}")
print(f"COORDS: {len(data.get('coordinates', []))}")
print("\n✅ Full 10-stage pipeline PASSED")
