# -*- coding: utf-8 -*-
"""Stress Testing — histórico · hipotético · reverse.

Thin wrapper sobre ``riskpkg.stress``. Tres pestañas:
1. Histórico  → ``historical_stress_test`` + ``historical_stress_battery``
2. Hipotético → ``apply_shock`` con ``PREDEFINED_SHOCKS`` o ``FactorShock`` a medida
3. Reverse   → ``reverse_stress_test`` + ``reverse_stress_curve``

Sin cálculos en la web. Los importes monetarios usan ``initial_value = 100_000 €``
para coincidir con el notebook ``06_stress_testing.py`` (oráculo).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from riskpkg import RiskVisualizer  # noqa: E402
from riskpkg.stress import HISTORICAL_SCENARIOS, PREDEFINED_SHOCKS  # noqa: E402

from components.cache import (  # noqa: E402
    analyze_level3,
    assert_demo_tickers_cached,
    assert_minimum_window,
    run_custom_shock,
    run_historical_battery,
    run_historical_stress,
    run_predefined_shock,
    run_reverse_curve,
    run_reverse_test,
    safe_live_call,
)
from components.formatting import num, pct, show_matplotlib  # noqa: E402
from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import get_config, has_config  # noqa: E402
from components.ticker_names import display_many  # noqa: E402

#: Valor inicial usado en el notebook 06 (oráculo).
INITIAL_VALUE = 100_000.0

st.set_page_config(page_title="Stress Testing", page_icon="🔥", layout="wide")
plt.close("all")
render_sidebar()

st.title("🔥 Stress Testing")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

cfg = get_config()

if len(cfg.tickers) < 2:
    st.warning("El stress testing necesita al menos 2 activos.", icon="⚠️")
    st.stop()
assert_minimum_window(cfg)
assert_demo_tickers_cached(cfg, additional=(cfg.benchmark,))

st.caption(
    f"Valor inicial de referencia: **{INITIAL_VALUE:,.0f} €** "
    f"(idéntico al notebook `06_stress_testing.py`)."
)

tab_hist, tab_hyp, tab_rev = st.tabs(
    ["📅 Histórico", "🎯 Hipotético", "🔄 Reverse"]
)

# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — Histórico
# ══════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.subheader("Stress test histórico")
    st.caption(
        "Reaplica ventanas reales de crisis a la cartera actual. Activos sin datos "
        "en la ventana se excluyen y los pesos se renormalizan sobre los disponibles."
    )

    scenario_options = {
        f"{s.name} ({s.start} → {s.end})": k for k, s in HISTORICAL_SCENARIOS.items()
    }
    label = st.selectbox(
        "Escenario",
        options=list(scenario_options.keys()),
        index=list(scenario_options.values()).index("covid_2020"),
    )
    scenario_key = scenario_options[label]

    with st.spinner(f"Cargando ventana '{scenario_key}'…"):
        res = safe_live_call(
            run_historical_stress,
            cfg,
            tickers=tuple(cfg.tickers),
            weights=tuple(cfg.weights),
            scenario_key=scenario_key,
            source=cfg.data_source,
            return_type=cfg.return_type,
            initial_value=INITIAL_VALUE,
        )

    st.markdown(f"**{res['scenario_name']}** — {res['scenario_description']}")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("P&L cartera", pct(res["portfolio_pnl_pct"]))
    h2.metric("Max DD", pct(res["max_drawdown_pct"]))
    h3.metric("Días al mínimo", str(res["days_to_trough"]))
    h4.metric("Recuperada en ventana", "✅ Sí" if res["recovered_within_window"] else "⚠️ No")

    h5, h6, h7, h8 = st.columns(4)
    h5.metric("Valor final", f"{res['portfolio_value_final']:,.0f} €")
    h6.metric("Mínimo absoluto", f"{res['portfolio_value_min']:,.0f} €")
    h7.metric("Sesiones", str(res["n_sessions"]))
    h8.metric("Activos usados", f"{len(res['tickers_used'])} / {len(cfg.tickers)}")

    if res["tickers_excluded"]:
        st.warning(
            "Excluidos por falta de datos en la ventana: "
            f"{display_many(res['tickers_excluded'], source=cfg.data_source)}",
            icon="ℹ️",
        )

    with st.expander("P&L por activo (sobre pesos renormalizados)"):
        df = pd.DataFrame(
            {
                "P&L": [res["asset_pnl_pct"][t] for t in res["tickers_used"]],
                "Peso efectivo": [
                    res["weights_effective"][t] for t in res["tickers_used"]
                ],
            },
            index=display_many(res["tickers_used"], source=cfg.data_source),
        )
        df.index.name = "Activo"
        st.dataframe(
            df.style.format({"P&L": "{:.2%}", "Peso efectivo": "{:.2%}"}),
            use_container_width=True,
        )

    st.markdown("**Trayectoria de la cartera durante el escenario**")
    show_matplotlib(RiskVisualizer.plot_historical_stress_path, res)

    # ── Batería completa ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Batería completa (los 8 escenarios)")
    with st.spinner("Ejecutando los 8 escenarios…"):
        battery = safe_live_call(
            run_historical_battery,
            cfg,
            tickers=tuple(cfg.tickers),
            weights=tuple(cfg.weights),
            source=cfg.data_source,
            return_type=cfg.return_type,
            initial_value=INITIAL_VALUE,
        )

    bat_view = battery.copy()
    for col in ("pnl_pct", "max_drawdown_pct"):
        if col in bat_view.columns:
            bat_view[col] = bat_view[col].map(
                lambda v: f"{v:.2%}" if pd.notna(v) else "—"
            )
    if "value_final" in bat_view.columns:
        bat_view["value_final"] = bat_view["value_final"].map(
            lambda v: f"{v:,.0f} €" if pd.notna(v) else "—"
        )
    st.dataframe(bat_view, use_container_width=True)

    show_matplotlib(RiskVisualizer.plot_historical_stress_battery, battery)

# ══════════════════════════════════════════════════════════════════════════
# Tab 2 — Hipotético
# ══════════════════════════════════════════════════════════════════════════
with tab_hyp:
    st.subheader("Stress test hipotético (EBA / CCAR / a medida)")
    st.caption(
        "Aplica shocks instantáneos por clase de activo, con posibilidad de "
        "override por ticker. No descarga datos."
    )

    ticker_to_class = cfg.ticker_to_class
    tt_tuple = tuple(ticker_to_class.items())

    mode = st.radio(
        "Modo",
        options=["Predefinido (EBA, CCAR, etc.)", "A medida (FactorShock)"],
        horizontal=True,
    )

    if mode.startswith("Predefinido"):
        shock_options = {s.name: k for k, s in PREDEFINED_SHOCKS.items()}
        label = st.selectbox("Shock predefinido", options=list(shock_options.keys()))
        shock_key = shock_options[label]
        st.caption(PREDEFINED_SHOCKS[shock_key].description)
        with st.spinner(f"Aplicando shock '{shock_key}'…"):
            res = safe_live_call(
                run_predefined_shock,
                cfg,
                tickers=tuple(cfg.tickers),
                weights=tuple(cfg.weights),
                shock_key=shock_key,
                ticker_to_class=tt_tuple,
                initial_value=INITIAL_VALUE,
            )
    else:
        st.markdown(
            "Define un `FactorShock` con shocks por **clase** (proporciones; "
            "p.ej. `-0.30` = -30%). Los *overrides por ticker* tienen prioridad."
        )
        name = st.text_input("Nombre", value="Crisis tech + flight-to-quality")
        description = st.text_area(
            "Descripción", value="Caída en tech y huida hacia oro.", height=80
        )
        c1, c2 = st.columns(2)
        with c1:
            sh_equity = st.number_input("Shock equity", value=-0.20, step=0.05, format="%.2f")
            sh_fixed = st.number_input("Shock fixed_income", value=0.0, step=0.05, format="%.2f")
            sh_real_estate = st.number_input("Shock real_estate", value=0.0, step=0.05, format="%.2f")
        with c2:
            sh_commodity = st.number_input("Shock commodity", value=0.18, step=0.05, format="%.2f")
            sh_alt = st.number_input("Shock alternative", value=0.0, step=0.05, format="%.2f")
            sh_cash = st.number_input("Shock cash", value=0.0, step=0.05, format="%.2f")
        default_shock = st.number_input("Shock por defecto", value=0.0, step=0.05, format="%.2f")

        # Overrides por ticker
        overrides_text = st.text_input(
            "Overrides por ticker (formato `TICKER:-0.35,OTRO:0.10`)",
            value="AMZN:-0.35,TTWO:-0.40",
            help="Aplica solo a los tickers que aparecen en la cartera actual.",
        )
        overrides = {}
        if overrides_text.strip():
            try:
                for pair in overrides_text.split(","):
                    t, v = pair.strip().split(":")
                    if t.strip() in cfg.tickers:
                        overrides[t.strip()] = float(v.strip())
            except ValueError:
                st.error("Formato de overrides incorrecto.", icon="❌")
                overrides = {}

        class_shocks = {
            "equity": sh_equity,
            "fixed_income": sh_fixed,
            "real_estate": sh_real_estate,
            "commodity": sh_commodity,
            "alternative": sh_alt,
            "cash": sh_cash,
        }
        with st.spinner("Aplicando shock a medida…"):
            res = safe_live_call(
                run_custom_shock,
                cfg,
                tickers=tuple(cfg.tickers),
                weights=tuple(cfg.weights),
                ticker_to_class=tt_tuple,
                name=name,
                class_shocks=tuple(class_shocks.items()),
                ticker_overrides=tuple(overrides.items()),
                default_shock=default_shock,
                description=description,
                initial_value=INITIAL_VALUE,
            )

    st.markdown(f"**{res['shock_name']}** — {res.get('shock_description', '')}")
    p1, p2, p3 = st.columns(3)
    p1.metric("P&L cartera", pct(res["portfolio_pnl_pct"]))
    p2.metric("P&L (€)", f"{res['portfolio_pnl_eur']:,.0f} €")
    p3.metric("Valor final", f"{res['portfolio_value_after']:,.0f} €")

    df = pd.DataFrame(
        {
            "Clase": [res["ticker_to_class"][t] for t in cfg.tickers],
            "Shock aplicado": [res["asset_shocks"][t] for t in cfg.tickers],
            "Contribución (€)": [res["asset_contributions"][t] for t in cfg.tickers],
        },
        index=display_many(cfg.tickers, source=cfg.data_source),
    )
    df.index.name = "Activo"
    st.dataframe(
        df.style.format({"Shock aplicado": "{:.2%}", "Contribución (€)": "{:,.0f} €"}),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — Reverse
# ══════════════════════════════════════════════════════════════════════════
with tab_rev:
    st.subheader("Reverse stress test (Mahalanobis-óptimo)")
    st.caption(
        "Pregunta inversa: dada una pérdida objetivo, ¿qué shock plausible la "
        "produce? Solución cerrada de Breuer & Csiszár (2013) bajo Σ histórica."
    )

    # Reverse requiere asset_returns + weights_effective del Nivel 3.
    with st.spinner("Cargando Nivel 3 (Σ histórica)…"):
        lvl3 = safe_live_call(
            analyze_level3,
            cfg,
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

    target_loss = st.slider(
        "Pérdida objetivo",
        min_value=0.05,
        max_value=0.50,
        value=0.15,
        step=0.05,
        format="%.0f%%",
    )
    # Streamlit interpreta el slider en su unidad; pero pasamos proporción directa.
    horizon = st.radio("Horizonte", options=["annual", "daily"], horizontal=True)

    with st.spinner("Resolviendo reverse stress…"):
        rev = run_reverse_test(
            asset_returns=lvl3.asset_returns,
            weights=lvl3.weights_effective,
            target_loss=target_loss,
            horizon=horizon,
        )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Pérdida objetivo", f"{rev['target_loss']:.0%}")
    r2.metric("Distancia Mahalanobis", num(rev["mahalanobis_distance"], decimals=4))
    r3.metric("z-score plausibilidad", num(rev["plausibility_zscore"], decimals=4))
    r4.metric(
        "Prob. plausibilidad",
        f"{rev['plausibility_prob']:.2%}",
        help="P(pérdida ≥ objetivo) bajo gaussiana multivariante con Σ histórica.",
    )

    shock_df = pd.DataFrame(
        {"Shock óptimo (s*)": rev["shock_vector"].values},
        index=display_many(rev["shock_vector"].index, source=cfg.data_source),
    )
    shock_df.index.name = "Activo"
    st.dataframe(
        shock_df.style.format({"Shock óptimo (s*)": "{:.4%}"}),
        use_container_width=True,
    )

    st.caption(
        f"σ_cartera ({horizon}) = {rev['sigma_portfolio']:.4f}. "
        f"Comprobación de pérdida sobre cartera = {rev['portfolio_loss_check']:.4%}."
    )

    # ── Curva 5% → 50% ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Curva de plausibilidad (5% → 50%)")
    with st.spinner("Trazando la curva…"):
        curve = run_reverse_curve(
            asset_returns=lvl3.asset_returns,
            weights=lvl3.weights_effective,
            losses=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
            horizon=horizon,
        )

    curve_view = curve.copy()
    curve_view["plausibility_prob"] = curve_view["plausibility_prob"].map(lambda v: f"{v:.2%}")
    for col in ("mahalanobis_distance", "sigma_portfolio"):
        if col in curve_view.columns:
            curve_view[col] = curve_view[col].map(lambda v: f"{v:.4f}")
    st.dataframe(curve_view, use_container_width=True)

    show_matplotlib(RiskVisualizer.plot_reverse_stress_curve, curve)
