"""
╔══════════════════════════════════════════════════════════════════╗
║  Digitalizer — 10-Stage Medical-Grade Digitization Server      ║
╚══════════════════════════════════════════════════════════════════╝
"""
print("DEBUG: main.py started", flush=True)
import io, base64, os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from processing.pipeline import run_pipeline

app = FastAPI(title="Digitalizer — 10-Stage Architecture")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def _enc(img):
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _signal_plot(time, raw, filtered, beats):
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), dpi=100,
                             facecolor="#0a0e17", sharex=True)
    for ax in axes:
        ax.set_facecolor("#111824")
        ax.tick_params(colors="#7d8590", labelsize=8)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color("#30363d")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    t = np.array(time)
    r = np.array(raw)
    f = np.array(filtered)

    axes[0].plot(t, r, color="#484f58", lw=0.5, alpha=0.5, label="Raw")
    axes[0].plot(t, f, color="#2ea043", lw=1.3, label="Filtered")
    axes[0].set_ylabel("mV", color="#e6edf3", fontsize=10)
    axes[0].set_title("Reconstructed ECG Signal — Raw vs Filtered",
                      color="#e6edf3", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8,
                   facecolor="#111824", edgecolor="#30363d", labelcolor="#e6edf3")

    axes[1].plot(t, f, color="#58a6ff", lw=1.2)
    colors = {"P": "#f0883e", "Q": "#d2a8ff", "R": "#ff7b72",
              "S": "#79c0ff", "T": "#7ee787"}
    for beat in beats:
        for wn, clr in colors.items():
            w = beat.get(wn)
            if w and w.get("time") is not None and w["idx"] < len(f):
                axes[1].plot(w["time"], f[w["idx"]], "o", color=clr,
                             ms=6, zorder=5)
                axes[1].annotate(wn, (w["time"], f[w["idx"]]),
                                 textcoords="offset points", xytext=(0, 12),
                                 fontsize=9, fontweight="bold", color=clr,
                                 ha="center")
    axes[1].set_xlabel("Time (s)", color="#e6edf3", fontsize=10)
    axes[1].set_ylabel("mV", color="#e6edf3", fontsize=10)
    axes[1].set_title("Annotated ECG — PQRST Detection",
                      color="#e6edf3", fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0a0e17")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _annotated_img(overlay, sig, beats):
    img = overlay.copy()
    orig_x = sig.get("original_x", sig.get("x_px", []))
    orig_y = sig.get("original_y", sig.get("y_px", []))
    if not orig_x:
        return img
    x_arr, y_arr = np.array(orig_x), np.array(orig_y)
    colours = {"P": (62, 136, 240), "Q": (168, 210, 255),
               "R": (114, 123, 255), "S": (192, 255, 121),
               "T": (135, 231, 126)}
    for beat in beats:
        for wn, bgr in colours.items():
            w = beat.get(wn)
            if w and w["idx"] < len(x_arr):
                cx, cy = int(x_arr[w["idx"]]), int(y_arr[w["idx"]])
                cv2.circle(img, (cx, cy), 7, bgr, -1)
                cv2.circle(img, (cx, cy), 9, bgr, 2)
                cv2.putText(img, wn, (cx - 6, cy - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
    return img


@app.get("/")
async def root():
    return FileResponse("static/index.html")


from fastapi import FastAPI, File, UploadFile, Form

@app.post("/api/extract")
async def extract(
    file: UploadFile = File(...),
    grid_suppression: bool = Form(True),
    wavelet_denoise: bool = Form(True),
    notch_filter: str = Form("50")
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400,
                            content={"error": "Invalid image file."})

    config = {
        "grid_suppression": grid_suppression,
        "wavelet_denoise": wavelet_denoise,
        "notch_filter": notch_filter,
    }

    # ── Run the full 10-stage pipeline ──
    result = run_pipeline(image, config=config)

    sig = result["signal"]
    proc = result["processed_signal"]
    waves = result["wave_detection"]
    clinical = result["clinical"]

    # Build annotated image
    annotated = _annotated_img(
        result["images"]["overlay"], sig, waves.get("beats", []))

    # Build signal plot
    sig_plot = _signal_plot(
        sig["time_s"], proc["raw"], proc["filtered"],
        waves.get("beats", []))

    # Encode images
    imgs = {}
    for k, v in result["images"].items():
        imgs[k] = _enc(v)
    imgs["annotated"] = _enc(annotated)

    # Build coordinates
    coords = []
    x_px = sig.get("x_px", [])
    y_px = sig.get("y_px", [])
    time_s = sig.get("time_s", [])
    voltage_mv = sig.get("voltage_mv", [])
    for i in range(len(x_px)):
        coords.append({
            "x": int(x_px[i]) if i < len(x_px) else 0,
            "y": round(float(y_px[i]), 2) if i < len(y_px) else 0,
            "time_s": round(float(time_s[i]), 5) if i < len(time_s) else 0,
            "voltage_mv": round(float(voltage_mv[i]), 4) if i < len(voltage_mv) else 0,
        })

    return JSONResponse(content={
        "status": "success",
        "quality": result["quality"],
        "images": imgs,
        "signal_plot": sig_plot,
        "coordinates": coords,
        "wave_detection": {
            "metrics": waves.get("metrics", {}),
            "beats": waves.get("beats", []),
            "detection_info": waves.get("detection_info", {}),
        },
        "clinical": clinical,
        "calibration": result.get("calibration", {}),
        "grid_info": result.get("grid_info", {}),
        "extraction_info": result.get("extraction_info", {}),
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
