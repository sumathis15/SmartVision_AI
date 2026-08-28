---
title: SmartVision AI
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
python_version: "3.10"
pinned: false
---

# SmartVision AI — Intelligent Multi-Class Object Recognition

A 25-class computer vision system on a curated [COCO 2017](https://cocodataset.org/) subset. Four ImageNet CNNs (VGG16, ResNet50, MobileNetV2, EfficientNetB0) handle single-object classification; YOLOv8 handles multi-object detection. Both are served through a Streamlit app.

Developed by [Sumathi S](https://www.linkedin.com/in/sumathisaravanan)

---

## Features

- Image classification with four transfer-learning models and optional ensemble
- YOLOv8 detection with bounding boxes, labels, and adjustable confidence
- Performance dashboard (accuracy, F1, top-5, confusion matrices, mAP)
- Optional webcam detection
- Deployment-ready Streamlit layout for local use and Hugging Face Spaces

## 25 classes

person, bicycle, car, motorcycle, airplane, bus, truck, traffic light, stop sign, bench, bird, cat, dog, horse, cow, elephant, bottle, cup, bowl, pizza, cake, chair, couch, potted plant, bed

## Results

| Model | Test accuracy | Macro F1 | Top-5 | Inference (ms) | Size |
|---|---|---|---|---|---|
| VGG16 | 66.9% | 0.669 | 91.2% | 27.6 | 254 MB |
| ResNet50 | 76.5% | 0.764 | 93.6% | 28.6 | 173 MB |
| MobileNetV2 | 68.5% | 0.685 | 92.0% | 46.7 | 28 MB |
| EfficientNetB0 | 76.5% | 0.763 | 94.1% | 99.8 | 41 MB |
| YOLOv8m (val) | mAP@0.5 76.3% | mAP@0.5:0.95 56.6% | — | 31 FPS | 50 MB |
| YOLOv8m (test) | mAP@0.5 75.8% | — | — | — | — |

Per-class tables and plots: [`reports/metrics.json`](reports/metrics.json), [`reports/figures/`](reports/figures/).

ResNet50 is used as the default classifier (same accuracy as EfficientNetB0, lower latency).

## Repository

```
Smart_Vision_AI/
├── Smartvision.ipynb          # dataset stream, EDA, crops, YOLO labels
├── notebooks/                 # CNN training, YOLO, comparison, exports
├── src/                       # shared config, models, pipeline
├── app.py                     # Streamlit home
├── pages/                     # Classification, Detection, Performance, Webcam, About
├── models/                    # *.keras, yolov8_best.pt (Git LFS)
├── reports/                   # metrics.json, figures
├── requirements.txt
└── packages.txt               # apt packages for Hugging Face Spaces
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Weights in `models/` and `reports/metrics.json` are included via Git LFS.

## Training (Google Colab, T4 GPU)

1. [`Smartvision.ipynb`](Smartvision.ipynb) — stream `detection-datasets/coco`, EDA, 70/15/15 splits, YOLO `data.yaml`.
2. [`notebooks/02_classification_training.ipynb`](notebooks/02_classification_training.ipynb) — four CNNs.
3. [`notebooks/03_yolo_training.ipynb`](notebooks/03_yolo_training.ipynb) — YOLOv8m fine-tune.
4. [`notebooks/04_comparison_pipeline.ipynb`](notebooks/04_comparison_pipeline.ipynb) — charts, TFLite, `metrics.json`.

Preprocessing is taken from the streamed subset (bbox format, crop-area cutoff, RGB conversion), not from generic COCO defaults.

## Hugging Face Spaces

This README includes Streamlit Space YAML (`app_file: app.py`). Create a Space, attach this GitHub repo, and use CPU hardware. If memory is tight, disable VGG16 on the Classification page. `packages.txt` provides `libgl1` and `libglib2.0-0` for OpenCV.

## License / data

Training data: [COCO](https://cocodataset.org/) via Hugging Face [`detection-datasets/coco`](https://huggingface.co/datasets/detection-datasets/coco).
