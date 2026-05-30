# -*- coding: utf-8 -*-
"""Nivel 1 — Análisis de un activo individual (esqueleto, Fase 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import has_config  # noqa: E402

st.set_page_config(page_title="Nivel 1 — Activo", page_icon="📈", layout="wide")
render_sidebar()

st.title("📈 Nivel 1 — Activo individual")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

st.info(
    "Página en construcción (Fase 1). Aquí irán: selector de activo del universo, "
    "métricas básicas (retorno, volatilidad, VaR/ES, Sharpe/Sortino, drawdown), "
    "comparación volatilidad histórica vs GARCH/GJR (mostrando el fallback si no "
    "converge) y panel EVT (paramétrico vs histórico vs EVT-POT).",
    icon="🚧",
)
