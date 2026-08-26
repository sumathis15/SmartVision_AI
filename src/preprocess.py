"""Data-driven preprocessing helpers. Thresholds are computed from the actual subset."""

from __future__ import annotations

import random
from typing import Iterable, Literal, Sequence

import numpy as np
from PIL import Image, ImageOps

BBoxFormat = Literal["xywh_px", "xyxy_px", "xywh_norm", "xyxy_norm"]


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert whatever mode the dataset actually used to RGB."""
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def detect_bbox_format(
    bboxes: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
) -> BBoxFormat:
    """Infer bbox convention from real samples instead of assuming COCO xywh."""
    if not bboxes:
        return "xywh_px"

    arr = np.asarray(bboxes, dtype=np.float64)
    max_val = float(np.nanmax(arr))
    normalized = max_val <= 1.5

    # xyxy if x2 > x1 and y2 > y1 for most boxes AND (x2-x1) looks like a coordinate not a width
    x1, y1, a, b = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    xyxy_votes = np.mean((a > x1) & (b > y1) & ((a - x1) < image_width * 1.05))
    # If third/fourth values are consistently smaller than first two, they are likely w/h
    wh_votes = np.mean((a < image_width) & (b < image_height) & (a < image_width * 0.95))

    if normalized:
        return "xyxy_norm" if xyxy_votes > 0.7 and not (wh_votes > 0.7) else "xywh_norm"
    return "xyxy_px" if xyxy_votes > 0.85 and np.mean(a > x1) > 0.85 else "xywh_px"


def to_xywh_pixels(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    fmt: BBoxFormat,
) -> tuple[float, float, float, float]:
    x0, y0, a, b = [float(v) for v in bbox]
    if fmt == "xywh_px":
        return x0, y0, a, b
    if fmt == "xyxy_px":
        return x0, y0, a - x0, b - y0
    if fmt == "xywh_norm":
        return x0 * image_width, y0 * image_height, a * image_width, b * image_height
    return (
        x0 * image_width,
        y0 * image_height,
        (a - x0) * image_width,
        (b - y0) * image_height,
    )


def clip_xywh(
    x: float,
    y: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Clip a box to the image. Returns None if the clipped box has no area."""
    x2 = min(image_width, x + w)
    y2 = min(image_height, y + h)
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    nw = x2 - x1
    nh = y2 - y1
    if nw < 1 or nh < 1:
        return None
    return x1, y1, nw, nh


def bbox_area(w: float, h: float) -> float:
    return max(0.0, w) * max(0.0, h)


def xywh_to_yolo(
    x: float,
    y: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    xc = (x + w / 2.0) / image_width
    yc = (y + h / 2.0) / image_height
    return (
        float(np.clip(xc, 0, 1)),
        float(np.clip(yc, 0, 1)),
        float(np.clip(w / image_width, 0, 1)),
        float(np.clip(h / image_height, 0, 1)),
    )


def compute_min_crop_area(areas: Iterable[float], percentile: float = 5.0) -> float:
    """Choose a tiny-object cutoff from the collected subset, not a hardcoded pixel value."""
    values = np.asarray(list(areas), dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def should_letterbox(aspect_ratios: Iterable[float], extreme_share_threshold: float = 0.25) -> bool:
    """Letterbox if a large share of crops are far from square."""
    ratios = np.asarray(list(aspect_ratios), dtype=np.float64)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return False
    extreme = (ratios < 0.5) | (ratios > 2.0)
    return float(np.mean(extreme)) >= extreme_share_threshold


def resize_crop(
    image: Image.Image,
    size: int = 224,
    letterbox: bool = False,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if not letterbox:
        return image.resize((size, size), Image.Resampling.LANCZOS)
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=fill)


def stratified_shuffle_split(
    items: list,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # Keep all items: remainder goes to test
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def observed_image_modes(modes: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mode in modes:
        counts[mode] = counts.get(mode, 0) + 1
    return counts
