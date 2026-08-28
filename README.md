---
title: SmartVision AI
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
tags:
  - streamlit
pinned: false
---

# SmartVision AI — Intelligent Multi-Class Object Recognition

A 25-class computer vision system on a curated [COCO 2017](https://cocodataset.org/) subset. Four ImageNet CNNs (VGG16, ResNet50, MobileNetV2, EfficientNetB0) handle single-object classification; YOLOv8 handles multi-object detection. Both are served through a Streamlit app.

Developed by [Sumathi S](https://www.linkedin.com/in/sumathisaravanan)

---

## Features

- Image classification with four transfer-learning models
- YOLOv8 detection with bounding boxes, labels, and adjustable confidence
- Performance dashboard (accuracy, F1, top-5, confusion matrices, mAP)
- Deployment-ready Streamlit layout for local use and Hugging Face Spaces

## 25 classes

person, bicycle, car, motorcycle, airplane, bus, truck, traffic light, stop sign, bench, bird, cat, dog, horse, cow, elephant, bottle, cup, bowl, pizza, cake, chair, couch, potted plant, bed

## Results

| Model | Test accuracy | Macro F1 | Top-5 | Inference (ms) | Size |
|---|---|---|---|---|---|
| VGG16 | 66.9% | 0.669 | 91.2% | 27.6 | 254 MB |
| ResNet50 | 76.5% | 0.764 | 93.6% | 28.6 | 173 MB |
| MobileNetV2 | 68.5% | 0.685 | 92.0% | 46.7 | 28 MB |
| EfficientNetB0 | 80.7% | 0.80 | — | 99.8 | 41 MB |
| YOLOv8m (val) | mAP@0.5 76.3% | mAP@0.5:0.95 56.6% | — | 31 FPS | 50 MB |
| YOLOv8m (test) | mAP@0.5 75.8% | — | — | — | — |

Per-class tables and plots: [`reports/metrics.json`](reports/metrics.json), [`reports/figures/`](reports/figures/).

EfficientNetB0 is the default classifier (highest test accuracy). ResNet50 is the faster alternative at 76.5%.

## Repository

```
Smart_Vision_AI/
├── Smartvision.ipynb          # dataset stream, EDA, crops, YOLO labels
├── notebooks/                 # 02 classification, 03 YOLO, 04 comparison
├── src/                       # shared config, models, pipeline
├── app.py                     # Streamlit home
├── pages/                     # Classification, Detection, Performance, Webcam, About
├── models/                    # *.keras, yolov8_best.pt (Git LFS)
├── reports/                   # metrics.json, figures
├── requirements.txt
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
streamlit run app.py
```

Weights in `models/` and `reports/metrics.json` are included via Git LFS.

## Training (Google Colab, T4 GPU)

1. [`Smartvision.ipynb`](Smartvision.ipynb) — stream `detection-datasets/coco`, EDA, 70/15/15 splits, YOLO `data.yaml`.
2. [`notebooks/02_classification_training.ipynb`](notebooks/02_classification_training.ipynb) — four CNNs.
3. [`notebooks/03_yolo_training.ipynb`](notebooks/03_yolo_training.ipynb) — YOLOv8m fine-tune.
4. [`notebooks/04_comparison_pipeline.ipynb`](notebooks/04_comparison_pipeline.ipynb) — charts, TFLite, `metrics.json`.

Preprocessing is taken from the streamed subset (bbox format, crop-area cutoff, RGB conversion), not from generic COCO defaults.

## Live demo (Streamlit Community Cloud)

The hosted app uses **TFLite + ONNX** (no TensorFlow/PyTorch) so it can install on Streamlit Cloud.

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → repository `sumathis15/SmartVision_AI`, branch `main`, main file `app.py`.
3. If an old failed app is stuck on `libglib2.0-0`, **delete it** and create a new one. Reboot alone often keeps the first failed log.
4. Wait for pip to finish (several minutes). A good log will **not** mention `libglib2.0-0` or `packages.txt`.

## Hugging Face Spaces

Hugging Face now requires a paid plan for Docker/Gradio Spaces. Static Spaces are free but cannot run this app. Use Streamlit Community Cloud for the live demo.

## License / data

Training data: [COCO](https://cocodataset.org/) via Hugging Face [`detection-datasets/coco`](https://huggingface.co/datasets/detection-datasets/coco).
