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

MODEL_ORDER = ("VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0")


def _grouped_score_chart(data: pd.DataFrame, model_order: list[str]) -> None:
    import altair as alt

    long = data.reset_index().melt("Model", var_name="Metric", value_name="Score")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=model_order, axis=alt.Axis(labelAngle=0, title=None, labelLimit=180)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color("Metric:N", legend=alt.Legend(orient="top")),
            xOffset="Metric:N",
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def _single_bar_chart(data: pd.DataFrame, model_order: list[str], y_title: str) -> None:
    import altair as alt

    col = data.columns[0]
    src = data.reset_index()
    chart = (
        alt.Chart(src)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=model_order, axis=alt.Axis(labelAngle=0, title=None, labelLimit=180)),
            y=alt.Y(f"{col}:Q", title=y_title),
            tooltip=["Model", alt.Tooltip(f"{col}:Q", format=".1f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

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
    df.index.name = "Model"
    df = df.rename(
        columns={
            "accuracy": "Accuracy",
            "precision_macro": "Precision",
            "recall_macro": "Recall",
            "f1_macro": "Macro F1",
            "top5_accuracy": "Top-5",
            "inference_ms": "Inference (ms)",
            "model_size_mb": "Size (MB)",
        }
    )
    model_order = [n for n in MODEL_ORDER if n in df.index]
    df = df.reindex(model_order)
    show = df.round(3).copy()
    if "Top-5" in show.columns:
        show["Top-5"] = df["Top-5"].map(lambda v: "—" if pd.isna(v) else f"{float(v):.3f}")
    st.dataframe(show, use_container_width=True)
    if not placeholder:
        numeric = df.apply(pd.to_numeric, errors="coerce")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Accuracy vs Macro F1")
            score_cols = [c for c in ("Accuracy", "Macro F1") if c in numeric.columns]
            if score_cols:
                _grouped_score_chart(numeric[score_cols], model_order)
        with c2:
            st.caption("Inference time")
            if "Inference (ms)" in numeric.columns:
                _single_bar_chart(numeric[["Inference (ms)"]], model_order, y_title="ms")
        st.caption("Best model (from training report): **" + str(metrics.get("best_classification_model", "—")) + "**")
        if "Top-5" in numeric.columns:
            missing = [n for n in model_order if pd.isna(numeric.loc[n, "Top-5"])]
            if missing:
                st.caption("Top-5 was not recorded for " + ", ".join(missing) + ".")
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
