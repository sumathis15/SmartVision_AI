"""SmartVision AI — Streamlit entry. Page labels live here so the sidebar never shows 'app'."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

home = st.Page("home.py", title="Home", icon=":material/home:", default=True)
classification = st.Page(
    "pages/1_Image_Classification.py",
    title="Image Classification",
    icon=":material/image_search:",
    url_path="classification",
)
detection = st.Page(
    "pages/2_Object_Detection.py",
    title="Object Detection",
    icon=":material/center_focus_strong:",
    url_path="detection",
)
performance = st.Page(
    "pages/3_Model_Performance.py",
    title="Model Performance",
    icon=":material/bar_chart:",
    url_path="performance",
)
about = st.Page(
    "pages/4_About.py",
    title="About",
    icon=":material/info:",
    url_path="about",
)

pg = st.navigation([home, classification, detection, performance, about])
pg.run()
