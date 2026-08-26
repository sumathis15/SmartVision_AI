"""Page 5 — Live webcam detection (bonus)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import DEFAULT_CONFIDENCE
from src.pipeline import detect_objects, draw_detections, yolo_available
from src.ui import inject_css

st.set_page_config(page_title="Webcam | SmartVision AI", layout="wide")
inject_css()
st.title("Live webcam detection")
st.caption(
    "Uses the browser camera (`st.camera_input`) so this page works on Hugging Face Spaces, "
    "not only on a local OpenCV webcam."
)

if not yolo_available():
    st.error("Missing `models/yolov8_best.pt`. Webcam detection needs the trained YOLO weights.")
    st.stop()

conf = st.slider("Confidence threshold", 0.10, 0.90, float(DEFAULT_CONFIDENCE), 0.05)
shot = st.camera_input("Allow camera access, then take a frame")

if shot is None:
    st.info("Grant camera permission and capture a frame to run YOLO.")
    st.stop()

try:
    image = Image.open(shot)
    image.load()
except Exception as exc:
    st.error(f"Could not read the camera frame: {exc}")
    st.stop()

t0 = time.perf_counter()
try:
    dets = detect_objects(image, confidence=conf)
    canvas = draw_detections(image, dets)
except Exception as exc:
    st.error(f"Detection failed: {exc}")
    st.stop()
elapsed_ms = (time.perf_counter() - t0) * 1000
fps = 1000.0 / elapsed_ms if elapsed_ms else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Latency", f"{elapsed_ms:.0f} ms")
c2.metric("FPS (this frame)", f"{fps:.1f}")
c3.metric("Objects", len(dets))

st.image(canvas, use_container_width=True)
if dets:
    st.dataframe(
        pd.DataFrame(
            [{"class": d["class"], "confidence": round(d["confidence"], 3)} for d in dets]
        ),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.warning("No objects above the threshold in this frame.")
