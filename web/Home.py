# -*- coding: utf-8 -*-
"""Home del front-end de riesgo multi-activo (Streamlit multipágina).

Arranque:
    streamlit run web/Home.py

La cartera se define una sola vez en la barra lateral (válida para todas las
páginas). El motor de cálculo es siempre ``riskpkg`` (ver docs/WEB_SPEC.md).
"""
import sys
from pathlib import Path

# La carpeta web/ debe estar en sys.path para importar `components` desde las páginas.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import riskpkg  # noqa: E402

from components.cache import analyze_level1  # noqa: E402
from components.data_availability import get_availability  # noqa: E402
from components.formatting import num, pct  # noqa: E402
from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import get_config, has_config  # noqa: E402
from components.ticker_names import display_name  # noqa: E402

st.set_page_config(
    page_title="Home · Riesgo Multi-Activo",
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
Esta aplicación es la **capa de presentación** del sistema. No realiza
cálculos financieros propios: orquesta el paquete `riskpkg` (motor verificado,
109 tests en verde) y muestra sus resultados. Define la cartera en la
**barra lateral** y recorre los cuatro niveles funcionales y el stress
testing en las páginas de la izquierda.
"""
)

# ══════════════════════════════════════════════════════════════════════════
# Bloque 1 — Tarjeta de configuración actual
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Configuración actual de la cartera")

if not has_config():
    st.info("Aún no hay cartera configurada. Edítala en la barra lateral.", icon="👈")
else:
    cfg = get_config()
    source_label = "Demo (offline)" if cfg.data_source == "demo" else "Live (yfinance)"
    n_business = int(pd.bdate_range(cfg.start_date, cfg.end_date).size)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos", str(len(cfg.tickers)))
    c2.metric("Benchmark", cfg.benchmark or "—")
    c3.metric("Fuente", source_label)
    c4.metric("Sesiones bursátiles", f"{n_business}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Inicio", cfg.start_iso)
    c6.metric("Fin", cfg.end_iso)
    c7.metric("Tipo de retorno", cfg.return_type)
    c8.metric("Tasa libre de riesgo", pct(cfg.risk_free_rate))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Monte Carlo · sims", f"{cfg.mc_sims:,}")
    c10.metric("Monte Carlo · días", f"{cfg.mc_days}")
    c11.metric("Activos no financ.", str(len(cfg.non_financial_assets)))
    c12.metric("Semilla determinista", str(riskpkg.utils.constants.RANDOM_SEED))

    with st.expander("Pesos por activo"):
        df_w = pd.DataFrame(
            {
                "Activo": [display_name(t, source=cfg.data_source) for t in cfg.tickers],
                "Ticker": list(cfg.tickers),
                "Peso": list(cfg.weights),
            }
        )
        st.dataframe(
            df_w.style.format({"Peso": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# Bloque 2 — Mapa de páginas
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Mapa de páginas ↔ secciones del sistema")
st.markdown(
    "Cada página orquesta una capa concreta del motor `riskpkg`. La columna "
    "**Sección memoria** referencia el apartado correspondiente del documento "
    "del TFG (los `§?` se rellenan con los números reales antes de la entrega)."
)
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
        "Sección memoria": ["§?", "§?", "§?", "§?", "§?"],
    }
)

# ══════════════════════════════════════════════════════════════════════════
# Bloque 3 — Mini-galería por activo
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Mini-galería de la cartera")
st.caption(
    "Resumen rápido de cada activo. Las métricas vienen del `full_report` "
    "que devuelve `Level1_AssetAnalyzer` en `riskpkg`; el gráfico muestra "
    "la evolución acumulada de 1€ en la ventana configurada."
)

if not has_config() or not cfg.tickers:
    st.info("Configura una cartera para ver la galería.", icon="ℹ️")
else:
    n_tickers = len(cfg.tickers)
    with st.spinner(f"Calculando {n_tickers} activos…"):
        cards = []
        for ticker in cfg.tickers:
            try:
                r = analyze_level1(
                    ticker=ticker,
                    start=cfg.start_iso,
                    end=cfg.end_iso,
                    source=cfg.data_source,
                    return_type=cfg.return_type,
                )
                cards.append((ticker, r))
            except Exception as exc:  # noqa: BLE001
                cards.append((ticker, exc))

    grid_cols = st.columns(min(2, n_tickers) if n_tickers > 0 else 1)
    for i, (ticker, payload) in enumerate(cards):
        with grid_cols[i % len(grid_cols)]:
            label = display_name(ticker, source=cfg.data_source)
            st.markdown(f"**{label}**")
            if isinstance(payload, Exception):
                st.warning(
                    f"No se pudo cargar `{ticker}`: {payload}",
                    icon="⚠️",
                )
                continue
            m = payload.metrics
            cum = (1.0 + payload.returns).cumprod()
            st.line_chart(cum, height=120)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Retorno anual", pct(m["annual_return"]))
            mc2.metric("Vol. anual", pct(m["volatility_ann"]))
            mc3.metric("Max DD", pct(m["max_drawdown"]))

# ══════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════
st.success(
    "Estado actual: **sistema completo**. Las cuatro capas funcionales "
    "(Niveles 1-4) y el módulo de stress testing están cableadas al motor "
    "`riskpkg` (v0.5.0, 109 tests en verde). Modo demo offline y modo live "
    "operativos.",
    icon="✅",
)
