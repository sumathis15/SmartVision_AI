"""Plotting helpers for EDA, confusion matrices, and model comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import CLASS_NAMES, FIGURES_DIR


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def plot_class_counts(counts: dict[str, int], title: str, filename: str):
    import matplotlib.pyplot as plt

    names = list(counts.keys())
    values = [counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(len(names)), values, color="#2E86AB")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_histogram(values, title: str, xlabel: str, filename: str, bins: int = 30):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(list(values), bins=bins, color="#2E86AB", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: np.ndarray, filename: str, title: str = "Confusion Matrix"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_model_comparison(rows: list[dict], metric_keys: list[str], filename: str):
    import matplotlib.pyplot as plt

    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    width = 0.8 / max(len(metric_keys), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(metric_keys):
        vals = [r.get(key, 0) for r in rows]
        ax.bar(x + i * width, vals, width, label=key)
    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Classification model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_cooccurrence(matrix: np.ndarray, filename: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matrix, cmap="YlOrRd")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_title("Class co-occurrence in collected detection images")
    fig.tight_layout()
    path = _save(fig, filename)
    plt.close(fig)
    return path


def figure_path(name: str) -> Path:
    return FIGURES_DIR / name
