# -*- coding: utf-8 -*-
"""Nivel 3 — Cartera diversificada + benchmark + IA.

Thin wrapper sobre ``Level3_PortfolioAnalyzer``. Cero cálculos en la web:
toda métrica, importancia y simulación viene tal cual de ``riskpkg``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from riskpkg import RiskVisualizer  # noqa: E402

from components.cache import analyze_level3  # noqa: E402
from components.formatting import num, pct, show_matplotlib  # noqa: E402
from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import get_config, has_config  # noqa: E402

#: Criterio TFG para coherencia RF vs MRC clásico (ver bloque 6 de la memoria).
SPEARMAN_THRESHOLD = 0.70

st.set_page_config(page_title="Nivel 3 — Cartera", page_icon="📊", layout="wide")
plt.close("all")
render_sidebar()

st.title("📊 Nivel 3 — Cartera diversificada")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

cfg = get_config()

if len(cfg.tickers) < 2:
    st.warning("La cartera necesita al menos 2 activos. Añade más tickers.", icon="⚠️")
    st.stop()

with st.spinner(f"Ejecutando Nivel 3 sobre '{cfg.portfolio_name}'…"):
    result = analyze_level3(
        tickers=tuple(cfg.tickers),
        weights=tuple(cfg.weights),
        start=cfg.start_iso,
        end=cfg.end_iso,
        source=cfg.data_source,
        return_type=cfg.return_type,
        benchmark=cfg.benchmark,
        portfolio_name=cfg.portfolio_name,
        mc_sims=cfg.mc_sims,
        mc_days=cfg.mc_days,
    )

m = result.port_metrics
tickers_eff = result.tickers_effective
weights_eff = result.weights_effective

# ── Aviso de exclusiones ────────────────────────────────────────────────────
excluded = [t for t in cfg.tickers if t not in tickers_eff]
if excluded:
    st.warning(
        f"`riskpkg` excluyó {excluded} por falta de datos en la ventana y renormalizó "
        f"los pesos. Cartera efectiva: {len(tickers_eff)} activos.",
        icon="ℹ️",
    )

# ══════════════════════════════════════════════════════════════════════════
# Bloque 1 — Métricas agregadas de la cartera
# ══════════════════════════════════════════════════════════════════════════
st.subheader(f"Métricas agregadas · {cfg.portfolio_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Retorno anual", pct(m["annual_return"]))
c2.metric("Volatilidad anual", pct(m["volatility_ann"]))
c3.metric("Sharpe", num(m["sharpe_ratio"], decimals=2))
c4.metric("Sortino", num(m["sortino_ratio"], decimals=2))

c5, c6, c7, c8 = st.columns(4)
c5.metric("VaR Hist. (95%)", num(m["var_hist_95"]))
c6.metric("VaR Hist. (99%)", num(m["var_hist_99"]))
c7.metric("Expected Shortfall", num(m["expected_shortfall"]))
c8.metric("Max Drawdown", pct(m["max_drawdown"]))

st.metric(
    "Ratio de Diversificación",
    num(result.div_ratio, decimals=4),
    help="Choueifaty & Coignard (2008). >1 indica beneficio.",
)

# ══════════════════════════════════════════════════════════════════════════
# Bloque 2 — Análisis vs benchmark
# ══════════════════════════════════════════════════════════════════════════
if result.alpha_beta is not None:
    st.subheader(f"Análisis vs benchmark ({result.benchmark})")
    alpha, beta = result.alpha_beta
    uc, dc = result.up_down_capture
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Alpha Jensen (anual)", pct(alpha))
    b2.metric("Beta", num(beta, decimals=4))
    b3.metric("R²", num(result.r_squared, decimals=4))
    b4.metric("Tracking Error", pct(result.tracking_error))
    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Information Ratio", num(result.info_ratio, decimals=4))
    b6.metric("Up Capture Ratio", f"{uc:.1f}%")
    b7.metric("Down Capture Ratio", f"{dc:.1f}%")
    if result.bench_metrics is not None:
        b8.metric(
            f"Sharpe {result.benchmark}",
            num(result.bench_metrics["sharpe_ratio"], decimals=2),
        )
else:
    st.info(
        f"El benchmark `{result.benchmark}` no estaba disponible en la ventana "
        "solicitada; se omite el análisis comparativo.",
        icon="ℹ️",
    )

# ══════════════════════════════════════════════════════════════════════════
# Bloque 3 — Kupiec
# ══════════════════════════════════════════════════════════════════════════
k = result.kupiec
if "interpretation" in k:
    st.subheader("Backtesting VaR — Test de Kupiec (95%)")
    st.write(k["interpretation"])
    k1, k2, k3 = st.columns(3)
    k1.metric("Excedencias", f"{k['exceedances']} / {k['n_test']}")
    k2.metric("p-valor", num(k["p_value"], decimals=4))
    k3.metric("VaR usado", num(k.get("var_used"), decimals=4))

# ══════════════════════════════════════════════════════════════════════════
# Bloque 4 — IA: Isolation Forest (anomalías)
# ══════════════════════════════════════════════════════════════════════════
n_anomalies = int((result.anomalies == -1).sum())
n_total = int(len(result.port_returns))
st.subheader("IA — Isolation Forest (días de estrés)")
a1, a2 = st.columns(2)
a1.metric("Días anómalos", f"{n_anomalies} / {n_total}")
a2.metric("% sesiones", f"{n_anomalies / n_total:.1%}")

# ══════════════════════════════════════════════════════════════════════════
# Bloque 5 — IA: Random Forest + Spearman ρ vs MRC (criterio TFG ρ > 0.70)
# ══════════════════════════════════════════════════════════════════════════
st.subheader("IA — Random Forest vs MRC clásico (coherencia)")

sp = result.spearman
rho = float(sp["spearman_correlation"])
p_val = float(sp["p_value"])
meets = bool(sp["meets_criterion"])

s1, s2, s3 = st.columns(3)
s1.metric("ρ Spearman", num(rho, decimals=4))
s2.metric("p-valor", num(p_val, decimals=4))
s3.metric("Criterio TFG (ρ > 0.70)", "✅ Cumple" if meets else "⚠️ No cumple")

if meets:
    st.success(
        f"Coherencia entre Random Forest y MRC clásico: ρ = {rho:.4f} > {SPEARMAN_THRESHOLD}. "
        f"{sp['interpretation']}",
        icon="✅",
    )
else:
    st.error(
        f"Coherencia insuficiente: ρ = {rho:.4f} ≤ {SPEARMAN_THRESHOLD}. "
        f"{sp['interpretation']}",
        icon="⚠️",
    )

rf_imp = result.rf_results["rf_importances"]
rf_df = pd.DataFrame(
    {
        "Importancia RF": [float(rf_imp[t]) for t in tickers_eff],
        "PRC clásico (%)": [float(result.prc[t]) for t in tickers_eff],
        "MRC clásico": [float(result.mrc[t]) for t in tickers_eff],
    },
    index=list(tickers_eff),
)
rf_df.index.name = "Ticker"
st.dataframe(
    rf_df.style.format(
        {"Importancia RF": "{:.4f}", "PRC clásico (%)": "{:.2f}%", "MRC clásico": "{:.4f}"}
    ),
    use_container_width=True,
)
st.caption(
    f"Random Forest entrenado con walk-forward TS-CV "
    f"({result.rf_results['n_cv_splits']} splits). "
    f"Ranking RF: {' > '.join(result.rf_results['rf_ranking'])}."
)

# ══════════════════════════════════════════════════════════════════════════
# Bloque 6 — Monte Carlo
# ══════════════════════════════════════════════════════════════════════════
mc = result.mc_results
st.subheader(f"Monte Carlo ({mc['n_sims']} sims, {mc['days']} días, seed={mc['seed']})")
mc_cols = st.columns(5)
mc_cols[0].metric("Valor inicial", f"{mc['initial_value']:,.0f} €")
mc_cols[1].metric("P5 (peor 5%)", f"{mc['percentiles'][5]:,.0f} €")
mc_cols[2].metric("Mediana P50", f"{mc['percentiles'][50]:,.0f} €")
mc_cols[3].metric("P95 (mejor 5%)", f"{mc['percentiles'][95]:,.0f} €")
mc_cols[4].metric("Prob. pérdida", f"{mc['prob_loss']:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# Bloque 7 — Figuras (orden y firmas idénticas al notebook 01)
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Visualizaciones")

st.markdown("**Retornos acumulados — cartera vs benchmark**")
show_matplotlib(
    RiskVisualizer.plot_cumulative_returns,
    result.asset_returns,
    title="Retornos Acumulados — Cartera vs Benchmark",
    benchmark_returns=result.bench_returns,
)

st.markdown("**Matriz de correlación**")
show_matplotlib(RiskVisualizer.plot_correlation_matrix, result.corr_matrix)

st.markdown("**Atribución del riesgo (MRC / PRC / importancia RF)**")
show_matplotlib(
    RiskVisualizer.plot_risk_attribution,
    result.mrc,
    result.prc,
    rf_importances=result.rf_results["rf_importances"],
)

st.markdown("**Anomalías detectadas (Isolation Forest)**")
show_matplotlib(RiskVisualizer.plot_anomalies, result.port_returns, result.anomalies)

st.markdown("**Simulación Monte Carlo**")
show_matplotlib(RiskVisualizer.plot_monte_carlo, result.mc_results)

if result.bench_returns is not None:
    st.markdown("**Rolling metrics (cartera vs benchmark)**")
    show_matplotlib(
        RiskVisualizer.plot_rolling_metrics,
        result.port_returns,
        result.bench_returns,
    )
