# -*- coding: utf-8 -*-
"""Nivel 1 — Análisis de un activo individual.

Thin wrapper sobre ``Level1_AssetAnalyzer``. Cero cálculos en la web:
métricas y figuras vienen tal cual de ``riskpkg``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
from riskpkg import RiskVisualizer  # noqa: E402

from components.cache import (  # noqa: E402
    analyze_level1,
    assert_demo_tickers_cached,
    assert_minimum_window,
    safe_live_call,
)
from components.formatting import metrics_to_table, num, pct, show_matplotlib  # noqa: E402
from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import get_config, has_config  # noqa: E402
from components.ticker_names import display_name  # noqa: E402

st.set_page_config(page_title="Nivel 1 — Activo", page_icon="📈", layout="wide")
plt.close("all")  # barrer cualquier figura residual antes de renderizar
render_sidebar()

st.title("📈 Nivel 1 — Activo individual")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

cfg = get_config()
assert_minimum_window(cfg)
assert_demo_tickers_cached(cfg)

# ── Selector de activo dentro del universo de la cartera ────────────────────
ticker = st.selectbox(
    "Activo a analizar",
    options=cfg.tickers,
    index=0,
    format_func=lambda t: display_name(t, source=cfg.data_source),
    help="Cada activo se analiza de forma aislada (sin pesos de cartera).",
)
ticker_label = display_name(ticker, source=cfg.data_source)

with st.spinner(f"Calculando métricas de {ticker_label}…"):
    result = safe_live_call(
        analyze_level1,
        cfg,
        ticker=ticker,
        start=cfg.start_iso,
        end=cfg.end_iso,
        source=cfg.data_source,
        return_type=cfg.return_type,
    )

m = result.metrics

# ── Bloque 0: evolución acumulada del activo ────────────────────────────────
st.subheader(f"Evolución de {ticker_label} · {cfg.start_iso} → {cfg.end_iso}")
st.caption(
    "Valor de 1€ invertido al inicio del período, compuesto con los "
    "retornos diarios. Misma serie sobre la que se calculan las métricas."
)
show_matplotlib(
    RiskVisualizer.plot_cumulative_returns,
    result.returns.to_frame(name=ticker_label),
    title=f"Evolución acumulada — {ticker_label}",
)

# ── Bloque 1: métricas básicas ──────────────────────────────────────────────
st.subheader(f"Métricas de {ticker_label} · {cfg.start_iso} → {cfg.end_iso}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Retorno anual", pct(m["annual_return"]))
c2.metric("Volatilidad anual", pct(m["volatility_ann"]))
c3.metric("Sharpe", num(m["sharpe_ratio"], decimals=2))
c4.metric("Sortino", num(m["sortino_ratio"], decimals=2))

c5, c6, c7, c8 = st.columns(4)
c5.metric("VaR Paramétrico (95%)", num(m["var_param_95"]))
c6.metric("VaR Paramétrico (99%)", num(m["var_param_99"]))
c7.metric("VaR Histórico (95%)", num(m["var_hist_95"]))
c8.metric("VaR Histórico (99%)", num(m["var_hist_99"]))

c9, c10, c11, c12 = st.columns(4)
c9.metric("Expected Shortfall", num(m["expected_shortfall"]))
c10.metric("Max Drawdown", pct(m["max_drawdown"]))
rec = m["recovery_days"]
c11.metric(
    "Recuperación",
    f"{rec} sesiones" if rec is not None else "Sin recuperar",
)
c12.metric("Sesiones analizadas", str(len(result.returns)))

# ── Bloque 2: tabla detallada ───────────────────────────────────────────────
with st.expander("Tabla completa de métricas (full_report)"):
    st.dataframe(
        metrics_to_table(
            m,
            [
                ("Etiqueta", "label", "raw"),
                ("Retorno anual", "annual_return", "pct"),
                ("Volatilidad anual", "volatility_ann", "pct"),
                ("VaR Paramétrico 95%", "var_param_95", "num"),
                ("VaR Paramétrico 99%", "var_param_99", "num"),
                ("VaR Histórico 95%", "var_hist_95", "num"),
                ("VaR Histórico 99%", "var_hist_99", "num"),
                ("Expected Shortfall", "expected_shortfall", "num"),
                ("Sharpe Ratio", "sharpe_ratio", "num"),
                ("Sortino Ratio", "sortino_ratio", "num"),
                ("Max Drawdown", "max_drawdown", "pct"),
                ("Recuperación (sesiones)", "recovery_days", "raw"),
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )

# ── Bloque 3: volatilidad condicional GARCH / GJR-GARCH ─────────────────────
st.subheader("Volatilidad condicional")
g, gjr = result.garch, result.gjr
g_ok, gjr_ok = bool(g.get("converged")), bool(gjr.get("converged"))

col_a, col_b, col_c = st.columns(3)
col_a.metric("Histórica anualizada", pct(m["volatility_ann"]))
col_b.metric(
    "GARCH(1,1)",
    pct(g["vol_forecast_ann"]) if g_ok else "Fallback",
    help=g.get("model", "—") if not g_ok else None,
)
col_c.metric(
    "GJR-GARCH",
    pct(gjr["vol_forecast_ann"]) if gjr_ok else "Fallback",
    help=gjr.get("model", "—") if not gjr_ok else None,
)
if not (g_ok and gjr_ok):
    st.caption(
        "Aviso: si un modelo no converge (o `arch` no está disponible), `riskpkg` "
        "cae a la volatilidad histórica como *fallback* documentado."
    )

# ── Bloque 4: backtesting VaR (Kupiec) ──────────────────────────────────────
k = result.kupiec
if "interpretation" in k:
    st.subheader("Backtesting VaR — Test de Kupiec (95%)")
    st.write(k["interpretation"])
    cols = st.columns(3)
    cols[0].metric("Excedencias observadas", f"{k['exceedances']} / {k['n_test']}")
    cols[1].metric("p-valor", num(k["p_value"], decimals=4))
    cols[2].metric("VaR usado", num(k.get("var_used"), decimals=4))

# ── Bloque 5: distribución de retornos + VaR ────────────────────────────────
st.subheader("Distribución de retornos y VaR")
show_matplotlib(RiskVisualizer.plot_var_distribution, result.returns, label=ticker_label)
