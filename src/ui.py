"""Shared Streamlit styling and paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import METRICS_PATH, PROJECT_ROOT


def ensure_project_on_path():
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def inject_css():
    import streamlit as st

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; max-width: 1200px;}
        .sv-hero {
            background: linear-gradient(135deg, #0f2c4c 0%, #1a5276 55%, #148f77 100%);
            color: #fff; padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
        }
        .sv-hero h1 {margin: 0 0 0.4rem 0; font-size: 2rem;}
        .sv-hero p {margin: 0; opacity: 0.92;}
        .sv-card {
            border: 1px solid #e6e9ef; border-radius: 14px; padding: 1rem 1.1rem;
            background: #fff; height: 100%;
        }
        .sv-warn {
            background: #fff4e5; border: 1px solid #f5c16c; border-radius: 10px;
            padding: 0.8rem 1rem; color: #6a4b00;
        }
        div[data-testid="stMetric"] {background: #f7fafc; border-radius: 12px; padding: 0.4rem 0.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_metrics() -> tuple[dict | None, bool]:
    """Returns (metrics, is_placeholder)."""
    if not METRICS_PATH.exists():
        return None, True
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return data, bool(data.get("placeholder", False))


def read_image_bytes(uploaded) -> bytes | None:
    if uploaded is None:
        return None
    data = uploaded.getvalue()
    if not data:
        return None
    return data


ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
