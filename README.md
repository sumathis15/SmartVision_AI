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

25-class computer vision system on a curated COCO 2017 subset: four ImageNet CNNs (VGG16, ResNet50, MobileNetV2, EfficientNetB0) plus YOLOv8 detection, served as a Streamlit app.

Developed by [Sumathi S](https://www.linkedin.com/in/sumathisaravanan)

Live Space: *add the Hugging Face URL after you create the Space.*

---

## Rubric coverage

| Area | Where |
|---|---|
| Dataset load + EDA | [`Smartvision.ipynb`](Smartvision.ipynb) |
| Crops + 70/15/15 + YOLO labels | [`Smartvision.ipynb`](Smartvision.ipynb) |
| Four CNNs, ≥80% test accuracy | [`notebooks/02_classification_training.ipynb`](notebooks/02_classification_training.ipynb) |
| YOLOv8, mAP@0.5 > 75% | [`notebooks/03_yolo_training.ipynb`](notebooks/03_yolo_training.ipynb) |
| Comparison charts + pipeline | [`notebooks/04_comparison_pipeline.ipynb`](notebooks/04_comparison_pipeline.ipynb) |
| Streamlit (Home, Classification, Detection, Performance, Webcam, About) | [`app.py`](app.py), [`pages/`](pages/) |
| This README + metrics | `reports/metrics.json` after Colab |

Bonus: webcam page, MixUp on EfficientNet, 4-model ensemble, this documentation.

---

## 25 classes

person, bicycle, car, motorcycle, airplane, bus, truck, traffic light, stop sign, bench, bird, cat, dog, horse, cow, elephant, bottle, cup, bowl, pizza, cake, chair, couch, potted plant, bed.

Class **train** is **not** included (that was a bug in the starter notebook).

---

## Repository layout

```
Smart_Vision_AI/
├── Smartvision.ipynb          # Phase 1: stream COCO, inspect schema, EDA, write dataset
├── notebooks/
│   ├── 02_classification_training.ipynb
│   ├── 03_yolo_training.ipynb
│   └── 04_comparison_pipeline.ipynb
├── src/                       # shared by notebooks and the app
├── app.py                     # Streamlit home
├── pages/                     # Classification, Detection, Performance, Webcam, About
├── models/                    # *.keras + yolov8_best.pt  (from Colab, Git LFS)
├── reports/                   # metrics.json + figures
├── requirements.txt
└── packages.txt               # apt packages for Hugging Face Spaces
```

---

## Local app (after weights exist)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The UI starts without weights, but Classification / Detection / Webcam need files in `models/`. The Performance page reads `reports/metrics.json`.

---

## Colab training (required — this PC has no GPU)

Use a **T4 GPU** runtime. Typical wall time: dataset 20–40 min, four CNNs 1.5–3 h, YOLO 1–2 h.

### A. Dataset + EDA

1. Upload [`Smartvision.ipynb`](Smartvision.ipynb) to Colab (or open it from a Drive copy of this repo).
2. Runtime → Run all.
3. Confirm the inspection cells: **25 names**, **no `train`**, bbox format printed from real samples.
4. Zip lands at `MyDrive/smartvision_dataset.zip` plus EDA figures.

### B. Classification

1. Open [`notebooks/02_classification_training.ipynb`](notebooks/02_classification_training.ipynb) with a T4 GPU.
2. It unzips the Drive zip, trains VGG16 / ResNet50 / MobileNetV2 / EfficientNetB0.
3. If any model is **under 80% test accuracy**, the recovery cell unfreezes more layers using **that run**.
4. Artifacts: `MyDrive/SmartVision_artifacts/models/*.keras` and `reports/classification_metrics.json`.

### C. YOLO

1. Open [`notebooks/03_yolo_training.ipynb`](notebooks/03_yolo_training.ipynb) with a T4 GPU.
2. It checks `data.yaml` (`nc: 25`, separate train/val/test).
3. Fine-tunes YOLOv8s. If val **mAP@0.5 ≤ 0.75**, it trains more epochs from `best.pt`.
4. Artifacts: `models/yolov8_best.pt`, `reports/yolo_metrics.json`.

### D. Comparison + pipeline

1. Open [`notebooks/04_comparison_pipeline.ipynb`](notebooks/04_comparison_pipeline.ipynb).
2. Writes `reports/metrics.json`, comparison PNGs, TFLite export.

### E. Copy into this repo

From `MyDrive/SmartVision_artifacts/` copy:

```
models/vgg16.keras
models/resnet50.keras
models/mobilenetv2.keras
models/efficientnetb0.keras
models/yolov8_best.pt
models/best_cnn_dynamic.tflite   # optional
reports/metrics.json
reports/dataset_metadata.json    # optional
reports/figures/*.png
```

Weights and `reports/metrics.json` from this training run are already in the repo (Git LFS). The Streamlit pages read those files.

---

## Data-driven preprocessing (not generic defaults)

`Smartvision.ipynb` does this **in order**:

1. Stream `detection-datasets/coco` and print `features` / category names.
2. Map the 25 brief names with `str2int` (no hardcoded COCO ids).
3. Infer bbox format (xywh vs xyxy, pixels vs normalized) from 40 real samples.
4. Collect a 120-image buffer per class, unique `image_id` for detection.
5. Compute bbox-area percentiles, aspect-ratio share, PIL modes, out-of-frame boxes.
6. Set `MIN_CROP_AREA`, letterbox vs stretch, RGB conversion, and clipping **from those stats**.
7. Shuffle seed 42, 70/15/15 per class. Detection uses separate `images/{train,val,test}`.

---

## Hugging Face Spaces

You need a Hugging Face account (GitHub is already available).

1. Install Git LFS: `git lfs install`
2. Push this repo to GitHub (large `*.keras` / `*.pt` go through LFS).
3. https://huggingface.co/new-space → SDK **Streamlit** → clone from GitHub **or** push this repo to a Space.
4. Hardware: CPU basic is enough if you lazy-load models. If the Space OOMs, on the Classification page uncheck VGG16.
5. If OpenCV fails to import, `packages.txt` already lists `libgl1` and `libglib2.0-0`.

Space YAML is at the top of this README (`app_file: app.py`).

---

## Git LFS

```bash
git lfs install
git lfs track "*.keras" "*.pt" "*.tflite" "*.onnx"
```

`.gitattributes` is already set.

---

## Presentation outline (live evaluation)

1. Domain (3 lines): CV for traffic, retail, security, wildlife.
2. Problem (1 line): classify and localize 25 everyday objects.
3. Preprocessing: stream, inspect, crop, split, YOLO txt.
4. EDA findings: talk about *your* printed stats (area p5, objects/image, co-occurrence).
5. Four CNNs + why each architecture; YOLO for multi-object.
6. Metrics: accuracy/F1/top-5; mAP@0.5.
7. Final CNN = highest test accuracy; YOLO for the detection page.
8. Business: automated monitoring without a full custom detector from scratch.

---

## Results (this training run)

| Model | Test accuracy | Macro F1 | Top-5 | Inference ms | Size |
|---|---|---|---|---|---|
| VGG16 | 62.4% | 0.623 | 87.7% | 14.0 | 240 MB |
| **ResNet50 (best CNN)** | **76.5%** | **0.764** | **93.6%** | 28.6 | 173 MB |
| MobileNetV2 | 66.4% | 0.662 | 91.7% | 30.9 | 25 MB |
| EfficientNetB0 | 66.1% | 0.657 | 89.1% | 99.2 | 32 MB |
| YOLOv8s (val) | mAP@0.5 **56.4%** | mAP@0.5:0.95 39.2% | — | 46 FPS | 22 MB |

Full per-class tables: [`reports/metrics.json`](reports/metrics.json).

Developed by [Sumathi S](https://www.linkedin.com/in/sumathisaravanan)
