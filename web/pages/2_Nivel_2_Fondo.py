# -*- coding: utf-8 -*-
"""Nivel 2 — Análisis de un fondo (cesta con pesos fijos) (esqueleto, Fase 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import has_config  # noqa: E402

st.set_page_config(page_title="Nivel 2 — Fondo", page_icon="🧺", layout="wide")
render_sidebar()

st.title("🧺 Nivel 2 — Fondo de inversión")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

st.info(
    "Página en construcción (Fase 1). Aquí irán: métricas ponderadas del fondo, "
    "ratio y beneficio de diversificación, contribución al riesgo MRC/PRC por "
    "activo y matriz de correlación.",
    icon="🚧",
)
