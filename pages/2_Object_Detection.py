"""Page 3 — YOLOv8 object detection with confidence slider and optional CNN verify."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import DEFAULT_CONFIDENCE
from src.pipeline import (
    detect_objects,
    draw_detections,
    keras_available,
    verify_crops_with_cnn,
    yolo_available,
)
from src.ui import ALLOWED_IMAGE_TYPES, inject_css, load_metrics

st.set_page_config(page_title="Object Detection | SmartVision AI", layout="wide")
inject_css()
st.title("Object Detection")
st.caption("YOLOv8 localizes every in-scope object. Raise the confidence threshold if you see duplicate boxes.")

if not yolo_available():
    st.error("Missing YOLO weights (`models/yolov8_best.onnx` or `models/yolov8_best.pt`). Fine-tune YOLOv8 in the Colab notebook and copy the files here.")
    st.stop()

metrics, _ = load_metrics()
best_cnn = (metrics or {}).get("best_classification_model", "EfficientNetB0")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    uploaded = st.file_uploader("Upload a scene image", type=ALLOWED_IMAGE_TYPES)
with c2:
    conf = st.slider("Confidence threshold", 0.10, 0.90, float(DEFAULT_CONFIDENCE), 0.05)
with c3:
    verify = st.checkbox("Verify crops with best CNN", value=False)
    if verify and not keras_available(best_cnn):
        st.caption(f"{best_cnn} weights missing — verification disabled.")
        verify = False

if uploaded is None:
    st.info("Upload a JPG/PNG that may contain several of the 25 classes.")
    st.stop()
if uploaded.size == 0:
    st.error("The uploaded file is empty.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except UnidentifiedImageError:
    st.error("That file is not a readable image.")
    st.stop()
except Exception as exc:
    st.error(f"Could not open the image: {exc}")
    st.stop()

try:
    with st.spinner("Running YOLOv8..."):
        dets = detect_objects(image, confidence=conf)
        if verify:
            dets = verify_crops_with_cnn(image, dets, model_name=best_cnn)
        canvas = draw_detections(image, dets, show_cnn=verify)
except Exception as exc:
    st.error(f"Detection failed: {exc}")
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.image(canvas, caption=f"{len(dets)} objects above {conf:.0%} confidence", use_container_width=True)
with right:
    st.metric("Detections", len(dets))
    if not dets:
        st.warning("Nothing passed the threshold. Lower the slider or try a clearer photo.")
    else:
        rows = []
        for d in dets:
            row = {"Class": d["class"], "Confidence": round(d["confidence"], 3)}
            if verify:
                row["CNN class"] = d.get("cnn_class")
                row["CNN confidence"] = None if d.get("cnn_confidence") is None else round(d["cnn_confidence"], 3)
                row["Agree"] = d.get("agree")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
