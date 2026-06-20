# -*- coding: utf-8 -*-
"""Global sidebar: the single portfolio builder.

Rendered on every page so the cartera is defined once and read everywhere via
``st.session_state``. Validates inputs and renormalizes weights to sum 1
*before* the config reaches any riskpkg analyzer (whose ``__init__`` asserts
``abs(sum(weights) - 1) < 1e-6``; see API_MAP.md §5).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .cache import ensure_data_source
from .data_availability import clear_live_cache, get_availability
from .state import (
    DEMO_TICKERS,
    DEMO_WEIGHTS,
    PortfolioConfig,
    get_config,
    set_config,
)
from .ticker_names import display_many, display_name

#: Mínimo de sesiones que riskpkg considera fiable (MIN_OBSERVATIONS).
_MIN_TRADING_DAYS = 252
#: Aprox. de sesiones por año natural (para avisar de rangos cortos).
_TRADING_DAYS_PER_CALENDAR_YEAR = 252 / 365.25

#: Horizontes Monte Carlo expuestos en la sidebar. Etiquetas alineadas con
#: los marcos regulatorios y académicos más comunes (Basilea, CCAR/EBA,
#: Hull/Jorion). 252 = 1 año bursátil es el default canónico.
_MC_DAYS_CHOICES: list[int] = [10, 21, 63, 126, 252, 504, 756]
_MC_DAYS_LABELS: dict[int, str] = {
    10:  "10 días — VaR Basel III (FRTB)",
    21:  "21 días — Mensual",
    63:  "63 días — Trimestral",
    126: "126 días — Semestral",
    252: "252 días — Anual (estándar)",
    504: "504 días — 2 años (CCAR/EBA)",
    756: "756 días — 3 años (planificación)",
}


def _parse_tickers(raw: str) -> list[str]:
    """Normaliza una cadena de tickers separados por coma."""
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def render_sidebar() -> PortfolioConfig:
    """Renderiza el constructor de cartera y devuelve la config persistida."""
    cfg = get_config()

    with st.sidebar:
        st.header("⚙️ Constructor de cartera")

        # ── Fuente de datos ───────────────────────────────────────────────
        source = st.radio(
            "Fuente de datos",
            options=["demo", "live"],
            index=0 if cfg.data_source == "demo" else 1,
            format_func=lambda s: "Demo (offline)" if s == "demo" else "Live (yfinance)",
            horizontal=True,
            help=(
                "Demo: series pre-descargadas, sin conexión y reproducibles. "
                "Live: descarga en tiempo real desde Yahoo Finance."
            ),
        )

        # ── Universo de activos ────────────────────────────────────────────
        st.subheader("Activos")
        if st.button("↺ Restaurar cartera demo", use_container_width=True):
            set_config(
                PortfolioConfig(
                    tickers=list(DEMO_TICKERS),
                    weights=list(DEMO_WEIGHTS),
                    start_date=cfg.start_date,
                    end_date=cfg.end_date,
                    data_source=source,
                )
            )
            st.rerun()

        raw_tickers = st.text_input(
            "Tickers (separados por coma)",
            value=", ".join(cfg.tickers),
            help="Ej.: ITX.MC, AMZN, TTWO, ANA, GLD",
        )
        tickers = _parse_tickers(raw_tickers)

        if "ANA" in tickers:
            ana_label = display_name("ANA", source=source)
            st.info(
                f"{ana_label} tiene datos disponibles únicamente entre 2018-07 "
                "y 2022-03; la ventana de análisis se ajusta automáticamente.",
                icon="ℹ️",
            )

        # ── Pesos (uno por ticker) ─────────────────────────────────────────
        st.subheader("Pesos")
        prev = dict(zip(cfg.tickers, cfg.weights))
        equal = 1.0 / len(tickers) if tickers else 0.0
        raw_weights: list[float] = []
        cols = st.columns(2)
        for i, ticker in enumerate(tickers):
            with cols[i % 2]:
                w = st.number_input(
                    display_name(ticker, source=source),
                    min_value=0.0,
                    max_value=1.0,
                    value=float(prev.get(ticker, equal)),
                    step=0.01,
                    format="%.2f",
                    key=f"w_{ticker}",
                )
                raw_weights.append(w)

        total = sum(raw_weights)
        if tickers:
            if abs(total - 1.0) < 1e-6:
                weights = raw_weights
            elif total > 0:
                weights = [w / total for w in raw_weights]
                st.caption(f"⚠️ Los pesos sumaban {total:.2f}; renormalizados a 1.00.")
            else:
                weights = [equal] * len(tickers)
                st.caption("⚠️ Pesos a cero; aplicado reparto equiponderado.")
        else:
            weights = []

        # ── Disponibilidad de datos por ticker ─────────────────────────────
        if tickers:
            _render_availability(tickers, source)

        # ── Parámetros de mercado ──────────────────────────────────────────
        st.subheader("Parámetros")
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input(
                "Inicio", value=cfg.start_date, min_value=date(1995, 1, 1)
            )
        with c2:
            end_date = st.date_input(
                "Fin", value=cfg.end_date, min_value=date(1995, 1, 1)
            )

        benchmark = st.text_input("Benchmark", value=cfg.benchmark, help="Ej.: SPY").strip().upper()
        risk_free_pct = st.number_input(
            "Tasa libre de riesgo (%)",
            min_value=0.0,
            max_value=20.0,
            value=cfg.risk_free_rate * 100,
            step=0.01,
            format="%.2f",
        )

        with st.expander("Avanzado"):
            return_type = st.selectbox(
                "Tipo de retorno",
                options=["log", "simple"],
                index=0 if cfg.return_type == "log" else 1,
            )
            mc_sims = st.number_input(
                "Monte Carlo — simulaciones (paths)",
                min_value=100, max_value=10000,
                value=cfg.mc_sims, step=100,
                help=(
                    "Número de trayectorias. Más sims = bandas de confianza "
                    "más estrechas en VaR-MC y percentiles P5/P95."
                ),
            )
            mc_days_default_idx = (
                _MC_DAYS_CHOICES.index(cfg.mc_days)
                if cfg.mc_days in _MC_DAYS_CHOICES
                else _MC_DAYS_CHOICES.index(252)
            )
            mc_days = st.selectbox(
                "Monte Carlo — horizonte",
                options=_MC_DAYS_CHOICES,
                index=mc_days_default_idx,
                format_func=lambda d: _MC_DAYS_LABELS[d],
                help=(
                    "Días de trading proyectados. 252 = 1 año (estándar "
                    "académico). Aumentar el horizonte NO mejora la "
                    "precisión de las colas; para colas más fiables sube el "
                    "número de simulaciones."
                ),
            )
            portfolio_name = st.text_input("Nombre de la cartera", value=cfg.portfolio_name)

        # ── Validaciones (avisos, no bloqueos salvo error duro) ────────────
        _render_validation(tickers, start_date, end_date)

    # Construir y persistir la config (fuera del bloque sidebar para claridad).
    new_cfg = PortfolioConfig(
        tickers=tickers,
        weights=weights,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        risk_free_rate=risk_free_pct / 100.0,
        data_source=source,
        return_type=return_type,
        portfolio_name=portfolio_name,
        mc_sims=int(mc_sims),
        mc_days=int(mc_days),
        non_financial_assets=cfg.non_financial_assets,  # se editan en Nivel 4
    )
    set_config(new_cfg)

    # Mantener el shim de datos sincronizado con la fuente elegida.
    ensure_data_source(source)
    return new_cfg


def render_config_summary() -> None:
    """Muestra una línea-resumen de la cartera activa (para cabecera de páginas)."""
    cfg = get_config()
    source_label = "Demo (offline)" if cfg.data_source == "demo" else "Live (yfinance)"
    tickers_str = (
        ", ".join(display_many(cfg.tickers, source=cfg.data_source)) or "—"
    )
    bench_str = display_name(cfg.benchmark, source=cfg.data_source) if cfg.benchmark else "—"
    st.caption(
        f"**Cartera:** {tickers_str}  ·  "
        f"**Benchmark:** {bench_str}  ·  "
        f"**Período:** {cfg.start_iso} → {cfg.end_iso}  ·  "
        f"**Fuente:** {source_label}"
    )


def _render_availability(tickers: list[str], source: str) -> None:
    """Tabla colapsable con la ventana disponible por ticker.

    Demo: lectura directa del parquet (gratis). Live: ``yfinance`` cacheado
    con ``@st.cache_data``; el botón "Refrescar" invalida el cache para
    forzar una nueva consulta sin reabrir la app.
    """
    with st.expander("📅 Disponibilidad de datos por activo", expanded=False):
        if source == "live":
            cols = st.columns([3, 1])
            cols[0].caption(
                "Yahoo Finance puede tardar unos segundos la primera vez. "
                "Los resultados quedan cacheados hasta que pulses Refrescar."
            )
            if cols[1].button("Refrescar", use_container_width=True):
                clear_live_cache()
                st.rerun()
        else:
            st.caption("Rangos leídos directamente del cache offline.")

        rows: list[dict[str, object]] = []
        for t in tickers:
            av = get_availability(t, source=source)
            rows.append(
                {
                    "Activo": t,
                    "Desde": av["first"].date().isoformat() if av else "—",
                    "Hasta": av["last"].date().isoformat() if av else "—",
                    "Sesiones": av["n_sessions"] if av else 0,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_validation(tickers: list[str], start_date: date, end_date: date) -> None:
    """Muestra avisos de validación en el sidebar."""
    if not tickers:
        st.error("Añade al menos un ticker para poder analizar.")
        return
    if len(tickers) < 2:
        st.info("Con un solo activo: solo el Nivel 1 es aplicable.")
    if end_date <= start_date:
        st.error("La fecha de fin debe ser posterior a la de inicio.")
        return
    approx_sessions = (end_date - start_date).days * _TRADING_DAYS_PER_CALENDAR_YEAR
    if approx_sessions < _MIN_TRADING_DAYS:
        st.warning(
            f"El rango cubre ~{approx_sessions:.0f} sesiones (< {_MIN_TRADING_DAYS}). "
            "riskpkg considera el período insuficiente para métricas fiables."
        )
