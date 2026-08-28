"""Page 4 — Model comparison dashboard from reports/metrics.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, FIGURES_DIR
from src.ui import inject_css, load_metrics

st.set_page_config(page_title="Model Performance | SmartVision AI", layout="wide")
inject_css()
st.title("Model Performance")

metrics, placeholder = load_metrics()
if metrics is None or placeholder:
    st.warning(
        "Showing the metrics schema only. After Colab training, replace "
        "`reports/metrics.json` and drop confusion-matrix PNGs into `reports/figures/`."
    )
    if metrics is None:
        st.stop()

clf = metrics.get("classification") or {}
yolo = metrics.get("yolo") or {}

st.subheader("Classification (test set)")
if clf:
    rows = []
    for name, m in clf.items():
        rows.append(
            {
                "model": name,
                "accuracy": m.get("accuracy"),
                "precision_macro": m.get("precision_macro"),
                "recall_macro": m.get("recall_macro"),
                "f1_macro": m.get("f1_macro"),
                "top5_accuracy": m.get("top5_accuracy"),
                "inference_ms": m.get("inference_ms"),
                "model_size_mb": m.get("model_size_mb"),
            }
        )
    df = pd.DataFrame(rows).set_index("model")
    st.dataframe(df.round(3), use_container_width=True)
    if not placeholder:
        c1, c2 = st.columns(2)
        acc_cols = [c for c in ("accuracy", "f1_macro", "top5_accuracy") if c in df.columns]
        numeric = df.apply(pd.to_numeric, errors="coerce")
        if acc_cols and numeric[acc_cols].notna().any().any():
            c1.bar_chart(numeric[acc_cols])
        if "inference_ms" in numeric.columns and numeric["inference_ms"].notna().any():
            c2.bar_chart(numeric[["inference_ms"]])
        st.caption("Best model (from training report): **" + str(metrics.get("best_classification_model", "—")) + "**")
        n_tests = {name: m.get("n_test") for name, m in clf.items() if m.get("n_test") is not None}
        if len(set(n_tests.values())) > 1:
            st.caption("Test set size: " + ", ".join(f"{k} n={v}" for k, v in n_tests.items()) + ".")
else:
    st.info("No classification metrics yet.")

st.subheader("Object Detection (YOLOv8)")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("mAP@0.5", "—" if yolo.get("map50") is None else f"{yolo['map50']:.3f}")
d2.metric("mAP@0.5:0.95", "—" if yolo.get("map50_95") is None else f"{yolo['map50_95']:.3f}")
d3.metric("Precision", "—" if yolo.get("precision") is None else f"{yolo['precision']:.3f}")
d4.metric("Recall", "—" if yolo.get("recall") is None else f"{yolo['recall']:.3f}")
d5.metric("FPS", "—" if yolo.get("fps") is None else f"{yolo['fps']:.1f}")
if yolo.get("meets_map50_floor") is True:
    st.caption("Validation mAP@0.5 is above 0.75.")

ap = yolo.get("per_class_ap50") or {}
if ap:
    st.markdown("**Per-class AP@0.5**")
    st.bar_chart(pd.Series(ap).reindex(CLASS_NAMES))

st.subheader("Confusion Matrices")
names = [
    ("vgg16", "VGG16"),
    ("resnet50", "ResNet50"),
    ("mobilenetv2", "MobileNetV2"),
    ("efficientnetb0", "EfficientNetB0"),
]
cols = st.columns(2)
for i, (stem, label) in enumerate(names):
    path = FIGURES_DIR / f"cm_{stem}.png"
    with cols[i % 2]:
        if path.exists():
            st.image(Image.open(path), caption=label, use_container_width=True)
        else:
            st.caption(f"{label}: figure not copied yet (`reports/figures/cm_{stem}.png`)")

cmp = FIGURES_DIR / "model_comparison.png"
trade = FIGURES_DIR / "accuracy_speed_tradeoff.png"
if cmp.exists() or trade.exists():
    st.subheader("Comparison Charts")
    a, b = st.columns(2)
    if cmp.exists():
        a.image(str(cmp), use_container_width=True)
    if trade.exists():
        b.image(str(trade), use_container_width=True)

st.subheader("Class-wise F1 (best CNN)")
best = metrics.get("best_classification_model")
if best and best in clf and clf[best].get("per_class"):
    pc = clf[best]["per_class"]
    f1s = {k: v.get("f1", 0) for k, v in pc.items()}
    st.bar_chart(pd.Series(f1s).reindex(CLASS_NAMES))
    st.dataframe(pd.DataFrame(pc).T, use_container_width=True)

decisions = metrics.get("dataset_decisions") or {}
if decisions:
    st.subheader("Dataset Filters (computed from this subset)")
    st.json(decisions)
