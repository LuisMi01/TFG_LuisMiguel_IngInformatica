# -*- coding: utf-8 -*-
"""Home del front-end de riesgo multi-activo (Streamlit multipágina).

Arranque:
    streamlit run web/app.py

La cartera se define una sola vez en la barra lateral (válida para todas las
páginas). El motor de cálculo es siempre ``riskpkg`` (ver docs/WEB_SPEC.md).
"""
import sys
from pathlib import Path

# La carpeta web/ debe estar en sys.path para importar `components` desde las páginas.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components.sidebar import render_config_summary, render_sidebar  # noqa: E402

st.set_page_config(
    page_title="Riesgo Multi-Activo · TFG",
    page_icon="📊",
    layout="wide",
)

render_sidebar()

st.title("📊 Análisis y Gestión del Riesgo Financiero — Carteras Multi-Activo")
st.caption(
    "Técnicas Cuantitativas e Inteligencia Artificial · "
    "TFG Grado en Ingeniería Informática 2025-2026 · Luis Miguel Urbez Villar"
)
render_config_summary()

st.markdown(
    """
Esta aplicación es la **capa de presentación** del sistema. No realiza cálculos
financieros propios: orquesta el paquete `riskpkg` (motor verificado, 109 tests
en verde) y muestra sus resultados. Define la cartera en la **barra lateral** y
recorre los cuatro niveles funcionales y el stress testing en las páginas de la
izquierda.
"""
)

st.subheader("Mapa de páginas ↔ secciones del sistema")
st.table(
    {
        "Página": [
            "Nivel 1 — Activo",
            "Nivel 2 — Fondo",
            "Nivel 3 — Cartera",
            "Nivel 4 — Patrimonio",
            "Stress Testing",
        ],
        "Agregación": [
            "Activo individual",
            "Fondo (cesta con pesos fijos)",
            "Cartera diversificada + benchmark + IA",
            "Patrimonio global (financiero + no financiero)",
            "Cartera (3 aproximaciones)",
        ],
        "Motor riskpkg": [
            "Level1_AssetAnalyzer + GARCH/GJR + EVT",
            "Level2_FundAnalyzer (MRC/PRC, diversificación)",
            "Level3_PortfolioAnalyzer (α/β, MC, Iso-Forest, RF)",
            "Level4_PatrimonyAnalyzer (clase/liquidez)",
            "stress: histórico · hipotético · reverse",
        ],
    }
)

st.info(
    "Estado actual: **Fase 0 completada** (andamiaje + cartera en sesión + modo "
    "demo offline). Las páginas de análisis se implementan en las Fases 1 y 2.",
    icon="🧱",
)
