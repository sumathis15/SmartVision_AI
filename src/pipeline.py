"""End-to-end inference: YOLO detection, optional CNN verification, drawing."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import (
    CLASS_NAMES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    IMAGE_SIZE,
    MODELS_DIR,
    NAME_TO_IDX,
    NUM_CLASSES,
    classification_split_dir,
    model_path,
    yolo_path,
)
from src.preprocess import ensure_rgb, resize_crop


def _font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=4)
def load_keras_model(name: str):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing weights for {name}: {path}")
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


@lru_cache(maxsize=1)
def load_yolo_model():
    path = yolo_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing YOLO weights: {path}")
    from ultralytics import YOLO

    return YOLO(str(path))


def keras_available(name: str) -> bool:
    return model_path(name).exists()


def yolo_available() -> bool:
    return yolo_path().exists()


def preprocess_for_keras(image: Image.Image) -> np.ndarray:
    image = ensure_rgb(image)
    image = resize_crop(image, IMAGE_SIZE, letterbox=False)
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, 0)


def classify_image(image: Image.Image, model_name: str, top_k: int = 5) -> list[dict]:
    model = load_keras_model(model_name)
    preds = model.predict(preprocess_for_keras(image), verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]
    return [
        {"class": CLASS_NAMES[int(i)], "index": int(i), "confidence": float(preds[int(i)])}
        for i in top_idx
    ]


def classify_all_models(
    image: Image.Image,
    top_k: int = 5,
    model_names: tuple[str, ...] | None = None,
) -> dict[str, list[dict]]:
    names = model_names or ("VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0")
    results = {}
    errors = {}
    for name in names:
        if not keras_available(name):
            errors[name] = f"Weights not found: {model_path(name).name}"
            continue
        try:
            results[name] = classify_image(image, name, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 — surface any load/predict failure in the UI
            errors[name] = str(exc)
    return {"predictions": results, "errors": errors}


def detect_objects(
    image: Image.Image,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> list[dict]:
    model = load_yolo_model()
    rgb = ensure_rgb(image)
    results = model.predict(
        source=np.asarray(rgb),
        conf=confidence,
        iou=iou,
        verbose=False,
    )
    detections = []
    if not results:
        return detections
    r0 = results[0]
    names = r0.names if hasattr(r0, "names") else {i: n for i, n in enumerate(CLASS_NAMES)}
    boxes = r0.boxes
    if boxes is None:
        return detections
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < NUM_CLASSES else str(cls_id))
        detections.append(
            {
                "xyxy": [float(v) for v in xyxy],
                "class_id": cls_id,
                "class": label,
                "confidence": conf,
            }
        )
    return detections


def verify_crops_with_cnn(
    image: Image.Image,
    detections: list[dict],
    model_name: str = "EfficientNetB0",
) -> list[dict]:
    """Optional Phase 4.1: re-classify each YOLO crop with the best CNN."""
    if not keras_available(model_name):
        return detections
    rgb = ensure_rgb(image)
    w, h = rgb.size
    enriched = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            enriched.append({**det, "cnn_class": None, "cnn_confidence": None})
            continue
        crop = rgb.crop((x1, y1, x2, y2))
        top = classify_image(crop, model_name, top_k=1)[0]
        enriched.append(
            {
                **det,
                "cnn_class": top["class"],
                "cnn_confidence": top["confidence"],
                "agree": top["class"] == det["class"],
            }
        )
    return enriched


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    show_cnn: bool = False,
) -> Image.Image:
    canvas = ensure_rgb(image).copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    palette = _palette(len(CLASS_NAMES))
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        idx = NAME_TO_IDX.get(det["class"], det.get("class_id", 0) % NUM_CLASSES)
        color = palette[idx]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class']} {det['confidence']:.2f}"
        if show_cnn and det.get("cnn_class"):
            mark = "✓" if det.get("agree") else "≠"
            label += f" | CNN {det['cnn_class']} {det['cnn_confidence']:.2f} {mark}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, max(0, y1 - 18)), label, fill=(255, 255, 255), font=font)
    return canvas


def _palette(n: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(0)
    colors = rng.integers(40, 230, size=(n, 3), dtype=np.int32)
    return [tuple(int(c) for c in row) for row in colors]


def sample_classification_images(n_per_class: int = 1) -> list[tuple[str, Path]]:
    """Pick a few on-disk crops for the Home page demo, if the dataset is present."""
    found = []
    for split in ("test", "val", "train"):
        root = classification_split_dir(split)
        if not root.exists():
            continue
        for class_name in CLASS_NAMES:
            folder = root / class_name
            if not folder.exists():
                continue
            files = sorted(folder.glob("*.jpg"))[:n_per_class]
            for f in files:
                found.append((class_name, f))
        if found:
            break
    return found


def clear_model_cache():
    load_keras_model.cache_clear()
    load_yolo_model.cache_clear()
