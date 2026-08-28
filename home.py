"""SmartVision AI — Home page."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, NUM_CLASSES, model_path
from src.pipeline import keras_available, sample_classification_images, yolo_available
from src.ui import DEVELOPER_NAME, LINKEDIN_URL, inject_css, load_metrics, running_on_cloud

st.set_page_config(
    page_title="Home | SmartVision AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

st.markdown(
    f"""
    <div class="sv-hero">
      <h1>SmartVision AI</h1>
      <p>Intelligent multi-class object recognition - 25 COCO classes, four transfer-learning CNNs, and YOLOv8 detection.<br>
      Developed by <a href="{LINKEDIN_URL}">{DEVELOPER_NAME}</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics, placeholder = load_metrics()

r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)
r1c1.metric("Classes", NUM_CLASSES)
if metrics and not placeholder:
    best = metrics.get("best_classification_model", "-")
    acc = metrics.get("classification", {}).get(best, {}).get("accuracy")
    r1c2.metric("Best CNN", best)
    r2c1.metric("CNN test accuracy", "-" if acc is None else f"{acc:.1%}")
    r2c2.metric("YOLO mAP@0.5", f"{metrics.get('yolo', {}).get('map50', 0):.1%}")
else:
    r1c2.metric("Best CNN", "pending training")
    r2c1.metric("CNN test accuracy", "-")
    r2c2.metric("YOLO mAP@0.5", "-")

if placeholder or metrics is None:
    st.markdown(
        '<div class="sv-warn"><b>Models / metrics are not in this checkout yet.</b> '
        "Run the Colab notebooks, then copy <code>models/</code> and <code>reports/</code> "
        "into the repo. Classification, detection, and the dashboard will light up automatically.</div>",
        unsafe_allow_html=True,
    )

if running_on_cloud():
    st.info(
        "This hosted demo runs **EfficientNetB0 (TFLite)** and **YOLOv8 (ONNX)**. "
        "VGG16, ResNet50, and MobileNetV2 need TensorFlow and run in the local application. "
        "The Performance page still shows all trained metrics."
    )

st.subheader("What SmartVision AI does")
a, b, c = st.columns(3)
with a:
    st.markdown("**Image Classification**")
    st.write("Upload a close-up of one object. Compare VGG16, ResNet50, MobileNetV2, and EfficientNetB0 side by side.")
with b:
    st.markdown("**Object Detection**")
    st.write("Upload a full scene. YOLOv8 draws boxes, labels, and scores. Confidence is adjustable; CNN verification on crops is optional.")
with c:
    st.markdown("**Model Performance**")
    st.write("Accuracy, precision, recall, F1, top-5, inference time, model size, confusion matrices, and YOLO mAP - all from the training reports.")

st.subheader("How to use")
st.markdown(
    """
1. Open **Image Classification** or **Object Detection** in the sidebar.
2. Upload a JPG or PNG.
3. Read top-5 scores or inspect boxes. Raise the detection threshold if you see duplicates.
4. Open **Model Performance** for the full comparison used in the report.
    """
)

st.subheader("25 classes")
chips = "".join(f'<span class="sv-chip">{name}</span>' for name in CLASS_NAMES)
st.markdown(f'<div class="sv-chips">{chips}</div>', unsafe_allow_html=True)

st.subheader("Sample crops")
samples = sample_classification_images(n_per_class=1)
if samples:
    cols = st.columns(5)
    for i, (name, path) in enumerate(samples[:10]):
        with cols[i % 5]:
            st.image(Image.open(path), caption=name, use_container_width=True)
else:
    st.caption("Sample crops appear here when `smartvision_dataset/classification/` is present. They are optional on the hosted demo.")

st.subheader("Model status")
cols = st.columns(5)
for col, name in zip(cols, ("VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0")):
    with col:
        if keras_available(name):
            note = '<span class="ok">Ready</span>'
        elif model_path(name).exists():
            note = '<span class="local">Local Keras only</span>'
        else:
            note = '<span class="missing">Weights missing</span>'
        st.markdown(f'<div class="sv-status"><b>{name}</b>{note}</div>', unsafe_allow_html=True)
with cols[4]:
    note = '<span class="ok">Ready</span>' if yolo_available() else '<span class="missing">Weights missing</span>'
    st.markdown(f'<div class="sv-status"><b>YOLOv8</b>{note}</div>', unsafe_allow_html=True)
