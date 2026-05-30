# -*- coding: utf-8 -*-
"""Nivel 4 — Patrimonio global (financiero + no financiero) (esqueleto, Fase 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import has_config  # noqa: E402

st.set_page_config(page_title="Nivel 4 — Patrimonio", page_icon="🏦", layout="wide")
render_sidebar()

st.title("🏦 Nivel 4 — Patrimonio global")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

st.info(
    "Página en construcción (Fase 2). Aquí irá el **formulario** para añadir "
    "activos no financieros (nombre, valor, clase, liquidez, volatilidad estimada, "
    "correlación con mercado), la consolidación patrimonial y la descomposición del "
    "riesgo por clase de activo y por liquidez. "
    "Nota MVP: la cartera financiera se valora en 1 M€ de referencia (hardcoded en riskpkg).",
    icon="🚧",
)
