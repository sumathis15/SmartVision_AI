"""SmartVision AI — Home (Streamlit entrypoint for local + Hugging Face Spaces)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, NUM_CLASSES
from src.pipeline import sample_classification_images, yolo_available, keras_available
from src.ui import inject_css, load_metrics

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

st.markdown(
    """
    <div class="sv-hero">
      <h1>SmartVision AI</h1>
      <p>Intelligent multi-class object recognition — 25 COCO classes, four transfer-learning CNNs, and YOLOv8 detection.<br>
      Developed by <a href="https://www.linkedin.com/in/sumathisaravanan" style="color:#fff;text-decoration:underline;">Sumathi S</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics, placeholder = load_metrics()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Classes", NUM_CLASSES)
if metrics and not placeholder:
    best = metrics.get("best_classification_model", "—")
    acc = metrics.get("classification", {}).get(best, {}).get("accuracy")
    c2.metric("Best CNN", best if acc is None else f"{best}")
    c3.metric("Best CNN accuracy", "—" if acc is None else f"{acc:.1%}")
    c4.metric("YOLO mAP@0.5", f"{metrics.get('yolo', {}).get('map50', 0):.1%}")
else:
    c2.metric("Best CNN", "pending training")
    c3.metric("Best CNN accuracy", "—")
    c4.metric("YOLO mAP@0.5", "—")

if placeholder or metrics is None:
    st.markdown(
        '<div class="sv-warn"><b>Models / metrics are not in this checkout yet.</b> '
        "Run the Colab notebooks, then copy <code>models/</code> and <code>reports/</code> "
        "into the repo. Classification, detection, and the dashboard will light up automatically.</div>",
        unsafe_allow_html=True,
    )

st.subheader("What this app does")
a, b, c = st.columns(3)
with a:
    st.markdown("**Image classification**")
    st.write("Upload a close-up of one object. Compare VGG16, ResNet50, MobileNetV2, and EfficientNetB0 side by side.")
with b:
    st.markdown("**Object detection**")
    st.write("Upload a full scene. YOLOv8 draws boxes, labels, and scores. Confidence is adjustable; CNN verification on crops is optional.")
with c:
    st.markdown("**Performance**")
    st.write("Accuracy, precision, recall, F1, top-5, inference time, model size, confusion matrices, and YOLO mAP — all from the training reports.")

st.subheader("How to use")
st.markdown(
    """
1. Open **Image Classification** or **Object Detection** in the sidebar.
2. Upload a JPG/PNG.
3. Read top-5 scores or inspect boxes. Raise the detection threshold if you see duplicates.
4. Open **Model Performance** for the full comparison used in the report.
    """
)

st.subheader("25 classes")
st.write(", ".join(CLASS_NAMES))

st.subheader("Sample crops")
samples = sample_classification_images(n_per_class=1)
if samples:
    cols = st.columns(5)
    for i, (name, path) in enumerate(samples[:10]):
        with cols[i % 5]:
            st.image(Image.open(path), caption=name, use_container_width=True)
else:
    st.info("Sample crops appear here after `smartvision_dataset/classification/` is present (optional for the live Space).")

st.subheader("Model status")
cols = st.columns(5)
for col, name in zip(cols, ("VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0")):
    col.write(f"**{name}**")
    col.write("ready" if keras_available(name) else "weights missing")
cols[4].write("**YOLOv8**")
cols[4].write("ready" if yolo_available() else "weights missing")
