"""Page 2 — Image Classification (all 4 CNNs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.pipeline import classify_all_models, keras_available
from src.ui import ALLOWED_IMAGE_TYPES, inject_css, load_metrics

st.set_page_config(page_title="Classification | SmartVision AI", layout="wide")
inject_css()
st.title("Image classification")
st.caption("Single-object crops work best (the models were trained on 224×224 boxed objects).")

metrics, _ = load_metrics()
best = None
if metrics:
    best = metrics.get("best_classification_model")

available = [n for n in ("VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0") if keras_available(n)]
if not available:
    st.error(
        "No classification weights found in `models/` "
        "(expected vgg16.keras, resnet50.keras, mobilenetv2.keras, efficientnetb0.keras). "
        "Train on Colab and copy the files here."
    )
    st.stop()

on_spaces = bool(os.environ.get("SPACE_ID"))
default_models = (
    [n for n in ("MobileNetV2", "EfficientNetB0", "ResNet50") if n in available]
    if on_spaces
    else available
)
if not default_models:
    default_models = available

selected = st.multiselect(
    "Models to run",
    options=available,
    default=default_models,
    help="On Hugging Face CPU Spaces, start with MobileNetV2 + EfficientNetB0 to stay under RAM limits. VGG16 is the largest.",
)
uploaded = st.file_uploader(
    "Upload an image",
    type=ALLOWED_IMAGE_TYPES,
    help="JPEG, PNG, WEBP, or BMP",
)

if not selected:
    st.warning("Select at least one model.")
    st.stop()

if uploaded is None:
    st.info("Upload an image to run the selected CNNs side by side.")
    st.stop()

if uploaded.size == 0:
    st.error("The uploaded file is empty.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except UnidentifiedImageError:
    st.error("That file is not a readable image. Try a JPG or PNG.")
    st.stop()
except Exception as exc:
    st.error(f"Could not open the image: {exc}")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.image(image, caption=uploaded.name, use_container_width=True)

with st.spinner("Running classification models..."):
    bundle = classify_all_models(image, top_k=5, model_names=tuple(selected))

preds = bundle["predictions"]
errors = bundle["errors"]
if errors:
    for name, msg in errors.items():
        st.warning(f"{name}: {msg}")
if not preds:
    st.error("Every model failed to run. Check TensorFlow install and the weight files.")
    st.stop()

with right:
    n = len(preds)
    cols = st.columns(n)
    for col, (name, top) in zip(cols, preds.items()):
        with col:
            st.subheader(name)
            if best == name:
                st.caption("Selected best (highest test accuracy)")
            st.metric("Top-1", f"{top[0]['class']}", f"{top[0]['confidence']:.1%}")
            chart = {row["class"]: row["confidence"] for row in top}
            st.bar_chart(chart, height=220)
