# -*- coding: utf-8 -*-
"""Nivel 2 — Fondo de inversión (cesta con pesos fijos).

Thin wrapper sobre ``Level2_FundAnalyzer``. Cero cálculos en la web:
métricas y figuras vienen tal cual de ``riskpkg``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from riskpkg import RiskVisualizer  # noqa: E402

from components.cache import (  # noqa: E402
    analyze_level2,
    assert_demo_tickers_cached,
    assert_minimum_window,
    safe_live_call,
)
from components.formatting import num, pct, show_matplotlib  # noqa: E402
from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import get_config, has_config  # noqa: E402

st.set_page_config(page_title="Nivel 2 — Fondo", page_icon="💼", layout="wide")
plt.close("all")
render_sidebar()

st.title("💼 Nivel 2 — Fondo de inversión")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

cfg = get_config()

if len(cfg.tickers) < 2:
    st.warning(
        "El análisis de fondo necesita al menos 2 activos. Añade más tickers en la "
        "barra lateral.",
        icon="⚠️",
    )
    st.stop()
assert_minimum_window(cfg)
assert_demo_tickers_cached(cfg)

with st.spinner(f"Analizando '{cfg.portfolio_name}'…"):
    result = safe_live_call(
        analyze_level2,
        cfg,
        tickers=tuple(cfg.tickers),
        weights=tuple(cfg.weights),
        start=cfg.start_iso,
        end=cfg.end_iso,
        source=cfg.data_source,
        return_type=cfg.return_type,
        fund_name=cfg.portfolio_name,
    )

m = result.fund_metrics
tickers_eff = result.tickers_effective
weights_eff = result.weights_effective

# ── Aviso si riskpkg excluyó algún activo ───────────────────────────────────
excluded = [t for t in cfg.tickers if t not in tickers_eff]
if excluded:
    st.warning(
        f"`riskpkg` excluyó {excluded} por falta de datos en la ventana solicitada y "
        f"renormalizó los pesos restantes. La cartera efectiva tiene "
        f"{len(tickers_eff)} activos.",
        icon="ℹ️",
    )

# ── Bloque 1: métricas agregadas del fondo ──────────────────────────────────
st.subheader(f"Métricas agregadas · {cfg.portfolio_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Retorno anual", pct(m["annual_return"]))
c2.metric("Volatilidad anual", pct(m["volatility_ann"]))
c3.metric("Sharpe", num(m["sharpe_ratio"], decimals=2))
c4.metric("Sortino", num(m["sortino_ratio"], decimals=2))

c5, c6, c7, c8 = st.columns(4)
c5.metric("VaR Paramétrico (95%)", num(m["var_param_95"]))
c6.metric("VaR Histórico (95%)", num(m["var_hist_95"]))
c7.metric("Expected Shortfall", num(m["expected_shortfall"]))
c8.metric("Max Drawdown", pct(m["max_drawdown"]))

# ── Bloque 2: diversificación ───────────────────────────────────────────────
st.subheader("Diversificación")
d1, d2 = st.columns(2)
d1.metric(
    "Ratio de Diversificación",
    num(result.div_ratio, decimals=4),
    help="Choueifaty & Coignard (2008). >1 indica beneficio de diversificación.",
)
d2.metric(
    "Beneficio de Diversificación",
    pct(result.div_benefit),
    help="Reducción de la volatilidad anual respecto a la combinación lineal de las volatilidades individuales.",
)

# ── Bloque 3: contribución al riesgo (MRC / PRC) ────────────────────────────
st.subheader("Contribución al riesgo por activo")
mrc_prc = pd.DataFrame(
    {
        "Peso efectivo": [weights_eff[i] for i in range(len(tickers_eff))],
        "MRC": [float(result.mrc[t]) for t in tickers_eff],
        "PRC (%)": [float(result.prc[t]) for t in tickers_eff],
    },
    index=list(tickers_eff),
)
mrc_prc.index.name = "Ticker"
st.dataframe(
    mrc_prc.style.format(
        {"Peso efectivo": "{:.2%}", "MRC": "{:.4f}", "PRC (%)": "{:.2f}%"}
    ),
    use_container_width=True,
)

# ── Bloque 4: métricas individuales de los activos ──────────────────────────
with st.expander("Métricas individuales de cada activo (full_report)"):
    rows = []
    for t in tickers_eff:
        am = result.asset_metrics[t]
        rows.append(
            {
                "Ticker": t,
                "Retorno anual": pct(am["annual_return"]),
                "Volatilidad anual": pct(am["volatility_ann"]),
                "VaR Hist. 95%": num(am["var_hist_95"]),
                "ES": num(am["expected_shortfall"]),
                "Sharpe": num(am["sharpe_ratio"], decimals=2),
                "Max DD": pct(am["max_drawdown"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── Bloque 5: figuras ───────────────────────────────────────────────────────
st.subheader("Atribución del riesgo")
show_matplotlib(RiskVisualizer.plot_risk_attribution, result.mrc, result.prc)

st.subheader("Matriz de correlación de activos")
show_matplotlib(RiskVisualizer.plot_correlation_matrix, result.asset_returns.corr())
