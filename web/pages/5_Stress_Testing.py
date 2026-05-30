# -*- coding: utf-8 -*-
"""Stress Testing — histórico · hipotético · reverse (esqueleto, Fase 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import has_config  # noqa: E402

st.set_page_config(page_title="Stress Testing", page_icon="🔥", layout="wide")
render_sidebar()

st.title("🔥 Stress Testing")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

st.info(
    "Página en construcción (Fase 2). Tres pestañas: **Histórico** (8 ventanas de "
    "crisis + batería comparativa), **Hipotético** (catálogo EBA/CCAR + FactorShock "
    "a medida) y **Reverse** (pérdida objetivo → shock Mahalanobis-óptimo + curva "
    "de plausibilidad).",
    icon="🚧",
)
