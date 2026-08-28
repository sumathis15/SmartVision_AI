"""Page 5 — About / documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import CLASS_NAMES
from src.ui import inject_css, load_metrics

st.set_page_config(page_title="About | SmartVision AI", layout="wide")
inject_css()
st.title("About SmartVision AI")

st.markdown(
    """
SmartVision AI is a 25-class computer vision system on a curated COCO 2017 subset.
It combines **transfer-learning image classification** (VGG16, ResNet50, MobileNetV2, EfficientNetB0)
with **YOLOv8 multi-object detection**, served through this Streamlit app.

The dataset is streamed from Hugging Face (`detection-datasets/coco`). Class IDs, bbox format,
and crop-area cutoffs are inspected from the stream rather than assumed.

Developed by [Sumathi S](https://www.linkedin.com/in/sumathisaravanan)
    """
)

st.subheader("Dataset")
st.markdown(
    f"""
- **Source:** COCO 2017 via `detection-datasets/coco`
- **Classes ({len(CLASS_NAMES)}):** {", ".join(CLASS_NAMES)}
- **Size:** 200 images per class after a quality crop filter (70% / 15% / 15%)
- **Classification inputs:** 224×224 RGB object crops with context padding
- **Detection inputs:** full scenes, YOLO txt labels, `nc: 25`
    """
)

metrics, placeholder = load_metrics()
if metrics and not placeholder:
    dec = metrics.get("dataset_decisions") or {}
    if dec:
        st.markdown("**Filters computed on this subset**")
        st.json(dec)

st.subheader("Model architectures")
st.markdown(
    """
| Model | Role | Setup |
|---|---|---|
| VGG16 | Classification | ImageNet weights, frozen conv base, dense + dropout head |
| ResNet50 | Classification | Last 20 layers unfrozen, global average pooling head |
| MobileNetV2 | Classification | Frozen base, compact head (speed) |
| EfficientNetB0 | Classification | ImageNet, batch-norm head, last 80 layers fine-tuned |
| YOLOv8m | Detection | COCO-pretrained, freeze-10 fine-tune on the 25-class subset |
    """
)

st.subheader("Technical stack")
st.markdown(
    """
Python, TensorFlow / Keras, Ultralytics YOLOv8, OpenCV, Pillow, pandas, scikit-learn,
Streamlit, Hugging Face Datasets / Spaces.
    """
)
