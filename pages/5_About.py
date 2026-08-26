"""Page 6 — About / documentation."""

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
SmartVision AI is a 25-class computer vision platform built on a curated COCO 2017 subset.
It combines **transfer-learning image classification** (VGG16, ResNet50, MobileNetV2, EfficientNetB0)
with **YOLOv8 multi-object detection**, then serves both through this Streamlit app.

The dataset is streamed from Hugging Face (`detection-datasets/coco`); class IDs, bbox format,
and crop-area cutoffs are **inspected from the actual stream** rather than assumed.
    """
)

st.subheader("Dataset")
st.markdown(
    f"""
- **Source:** COCO 2017 via `detection-datasets/coco`
- **Classes ({len(CLASS_NAMES)}):** {", ".join(CLASS_NAMES)}
- **Target size:** 100 images per class (70% / 15% / 15% shuffled splits)
- **Classification inputs:** 224×224 RGB object crops
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
| Model | Role | Brief setup |
|---|---|---|
| VGG16 | Classification | ImageNet weights, frozen conv base, dense + dropout head |
| ResNet50 | Classification | Last 20 layers unfrozen, global average pooling head |
| MobileNetV2 | Classification | Frozen base, compact head (speed) |
| EfficientNetB0 | Classification | Mixed precision, batch-norm head, MixUp |
| YOLOv8s | Detection | COCO-pretrained, fine-tuned on the 25-class subset |
    """
)

st.subheader("Technical stack")
st.markdown(
    """
Python, TensorFlow / Keras, Ultralytics YOLOv8, OpenCV, Pillow, pandas, scikit-learn,
Streamlit, Hugging Face Datasets / Spaces.
    """
)

st.subheader("Developer")
st.markdown(
    """
**Name:** `[Your Name]`  
**Program:** `[Your program / institution]`  
**GitHub:** `[your-github-username]`

Replace the placeholders above before the live evaluation.
    """
)

st.subheader("Live evaluation talking points")
st.markdown(
    """
1. Domain: computer vision for cities, retail, security, wildlife, and similar verticals.
2. Problem: detect and classify 25 everyday object categories in real scenes.
3. Preprocessing: stream COCO, inspect schema, crop boxes, shuffled 70/15/15, YOLO labels.
4. EDA: class counts, objects/image, bbox area, co-occurrence — filters taken from those stats.
5. Models: four ImageNet CNNs plus YOLOv8s; pick the CNN on accuracy–speed, YOLO for localization.
6. Metrics: accuracy / F1 / top-5 for CNNs; mAP@0.5 and mAP@0.5:0.95 for YOLO.
7. Deployment: this Streamlit app on Hugging Face Spaces with lazy-loaded weights.
    """
)
