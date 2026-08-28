"""Shared Streamlit styling, sidebar credit, and paths."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.config import METRICS_PATH, PROJECT_ROOT

DEVELOPER_NAME = "Sumathi S"
LINKEDIN_URL = "https://www.linkedin.com/in/sumathisaravanan"

_LINKEDIN_ICON = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path fill="#0A66C2" d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>"""


def ensure_project_on_path():
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def inject_css():
    import streamlit as st

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
        }}
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2.4rem;
            max-width: 1180px;
        }}
        h1, h2, h3 {{ letter-spacing: -0.02em; color: #0b1f33; }}

        [data-testid="stSidebar"] {{
            background: #0b1f33;
        }}
        [data-testid="stSidebar"] * {{
            color: #e8eef5 !important;
        }}
        [data-testid="stSidebarNav"] {{
            padding-top: 0.4rem;
        }}
        [data-testid="stSidebarNav"] a {{
            border-radius: 10px;
            margin: 0.15rem 0.35rem;
            padding: 0.45rem 0.7rem !important;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(255,255,255,0.08);
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(14, 124, 102, 0.40);
        }}

        .sv-hero {{
            background: linear-gradient(135deg, #0b1f33 0%, #163a5f 52%, #0e7c66 100%);
            color: #fff;
            padding: 1.7rem 1.9rem;
            border-radius: 18px;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 28px rgba(11, 31, 51, 0.18);
        }}
        .sv-hero h1 {{ margin: 0 0 0.45rem 0; font-size: 2.05rem; color: #fff; }}
        .sv-hero p {{ margin: 0; opacity: 0.93; line-height: 1.5; }}
        .sv-hero a {{ color: #d7fff4; text-decoration: underline; text-underline-offset: 3px; }}

        .sv-card {{
            border: 1px solid #e4eaf1;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: #fff;
            height: 100%;
            box-shadow: 0 1px 2px rgba(11, 31, 51, 0.04);
        }}
        .sv-warn {{
            background: #fff7ea;
            border: 1px solid #f0c987;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            color: #6a4b00;
        }}
        .sv-chips {{
            display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.35rem;
        }}
        .sv-chip {{
            background: #eef4f8;
            color: #0b1f33;
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 500;
            border: 1px solid #d7e2ea;
        }}
        .sv-status {{
            border: 1px solid #e4eaf1; border-radius: 12px; padding: 0.7rem 0.8rem;
            background: #f8fbfd; text-align: center;
        }}
        .sv-status b {{ display: block; margin-bottom: 0.2rem; }}
        .sv-status .ok {{ color: #0e7c66; font-weight: 600; }}
        .sv-status .local {{ color: #6a4b00; font-size: 0.82rem; }}
        .sv-status .missing {{ color: #9b2c2c; font-size: 0.82rem; }}

        .sv-credit {{
            margin: 1.4rem 0.5rem 0.6rem;
            padding: 0.85rem 0.9rem 0.95rem;
            border-top: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }}
        .sv-credit-kicker {{
            display: block;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #9db0c4 !important;
            margin-bottom: 0.4rem;
        }}
        .sv-credit a {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: #ffffff !important;
            text-decoration: none !important;
            font-weight: 600;
            font-size: 0.92rem;
        }}
        .sv-credit a:hover {{ text-decoration: underline !important; }}
        .sv-li {{
            width: 22px; height: 22px; border-radius: 4px;
            background: #fff;
            display: inline-flex; align-items: center; justify-content: center;
            flex-shrink: 0;
        }}

        div[data-testid="stMetric"] {{
            background: #f6fafc;
            border: 1px solid #e4eaf1;
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.35rem;
            overflow: visible;
            white-space: nowrap;
        }}
        div[data-testid="stMetricLabel"] p {{
            font-weight: 600;
            color: #4a6278;
        }}
        .stDeployButton {{ display: none; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _sidebar_credit()


def _sidebar_credit():
    import streamlit as st

    st.sidebar.markdown(
        f"""
        <div class="sv-credit">
          <span class="sv-credit-kicker">Developed by</span>
          <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">
            <span class="sv-li">{_LINKEDIN_ICON}</span>
            {DEVELOPER_NAME}
          </a>
        </div>
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


def running_on_cloud() -> bool:
    """Hugging Face Spaces or Streamlit Community Cloud (tight RAM)."""
    if os.environ.get("SPACE_ID"):
        return True
    if os.path.exists("/mount/src"):
        return True
    return False
