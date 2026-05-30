# -*- coding: utf-8 -*-
"""Nivel 3 — Cartera diversificada + benchmark + IA (esqueleto, Fase 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import has_config  # noqa: E402

st.set_page_config(page_title="Nivel 3 — Cartera", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 Nivel 3 — Cartera diversificada")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

st.info(
    "Página en construcción (Fase 1) — la vitrina más completa. Aquí irán: análisis "
    "vs benchmark (alpha, beta, R², tracking error, IR, capture ratios), Monte Carlo "
    "con bandas de percentiles, Isolation Forest (anomalías) y Random Forest "
    "(importancia de factores + coherencia Spearman con MRC, ρ > 0,70).",
    icon="🚧",
)
