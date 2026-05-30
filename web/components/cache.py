# -*- coding: utf-8 -*-
"""Cached wrappers over riskpkg data access + level analyzers.

``@st.cache_data`` is safe here because riskpkg is deterministic
(``RANDOM_SEED = 42`` is fixed on import; Monte Carlo / Isolation Forest reseed
internally). Cache keys include ``data_source`` so demo and live never collide.

The wrappers extract serializable attributes from the analyzers (dicts,
``pd.Series``, ``pd.DataFrame``) — never the analyzer object itself — so cache
storage stays stable and the web layer adds zero financial calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st
from riskpkg.data import DataLoader
from riskpkg.levels import (
    Level1_AssetAnalyzer,
    Level2_FundAnalyzer,
    Level3_PortfolioAnalyzer,
    Level4_PatrimonyAnalyzer,
)
from riskpkg.stress import (
    FactorShock,
    apply_shock,
    historical_stress_battery,
    historical_stress_test,
    reverse_stress_curve,
    reverse_stress_test,
)

from . import demo_data


def ensure_data_source(source: str) -> None:
    """Sincroniza el shim global con la fuente de datos seleccionada.

    Debe llamarse antes de instanciar cualquier analizador de riskpkg, porque
    estos descargan datos por su cuenta (ver demo_data.py).
    """
    if source == "demo":
        demo_data.activate()
    else:
        demo_data.deactivate()


@st.cache_data(show_spinner=False)
def load_prices(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    source: str,
    return_type: str = "log",
) -> pd.DataFrame:
    """Precios de cierre vía ``DataLoader``. En demo, el shim evita la red."""
    ensure_data_source(source)
    loader = DataLoader(list(tickers), start, end, return_type=return_type).fetch()
    return loader.prices


@st.cache_data(show_spinner=False)
def load_returns(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    source: str,
    return_type: str = "log",
) -> pd.DataFrame:
    """Retornos (log o simple) vía ``DataLoader``."""
    ensure_data_source(source)
    loader = DataLoader(list(tickers), start, end, return_type=return_type).fetch()
    return loader.returns


# ── Resultados serializables de los analizadores ────────────────────────────


@dataclass
class Level1Result:
    """Foto serializable de ``Level1_AssetAnalyzer`` tras ``run()``.

    Espejo de los atributos públicos/privados que consumen los notebooks
    (ver docs/API_MAP.md §5).
    """

    ticker: str
    metrics: dict
    returns: pd.Series
    garch: dict
    gjr: dict
    kupiec: dict


@dataclass
class Level2Result:
    """Foto serializable de ``Level2_FundAnalyzer`` tras ``run()``."""

    fund_name: str
    fund_metrics: dict
    asset_metrics: dict
    fund_returns: pd.Series
    asset_returns: pd.DataFrame
    mrc: pd.Series
    prc: pd.Series
    div_ratio: float
    div_benefit: float
    weights_effective: tuple[float, ...]
    tickers_effective: tuple[str, ...]


@st.cache_data(show_spinner=False)
def analyze_level1(
    ticker: str,
    start: str,
    end: str,
    source: str,
    return_type: str = "log",
) -> Level1Result:
    """Ejecuta ``Level1_AssetAnalyzer.run()`` y devuelve sus atributos."""
    ensure_data_source(source)
    lvl1 = Level1_AssetAnalyzer(ticker, start, end, return_type=return_type).run()
    return Level1Result(
        ticker=ticker,
        metrics=lvl1.metrics,
        returns=lvl1.returns,
        garch=lvl1._garch or {},
        gjr=lvl1._gjr or {},
        kupiec=lvl1._kupiec or {},
    )


@st.cache_data(show_spinner=False)
def analyze_level2(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    start: str,
    end: str,
    source: str,
    return_type: str = "log",
    fund_name: str = "Fondo",
) -> Level2Result:
    """Ejecuta ``Level2_FundAnalyzer.run()`` y devuelve sus atributos.

    El sidebar ya renormaliza pesos a 1.0 antes de instanciar la config, así que
    el ``assert`` interno de riskpkg pasa siempre.
    """
    ensure_data_source(source)
    lvl2 = Level2_FundAnalyzer(
        tickers=list(tickers),
        weights=list(weights),
        fund_name=fund_name,
        start_date=start,
        end_date=end,
        return_type=return_type,
    ).run()
    tickers_eff = tuple(lvl2._asset_returns.columns.tolist())
    return Level2Result(
        fund_name=fund_name,
        fund_metrics=lvl2._fund_metrics,
        asset_metrics=lvl2._asset_metrics,
        fund_returns=lvl2._fund_returns,
        asset_returns=lvl2._asset_returns,
        mrc=lvl2._mrc,
        prc=lvl2._prc,
        div_ratio=float(lvl2._div_ratio),
        div_benefit=float(lvl2._div_benefit),
        weights_effective=tuple(float(w) for w in lvl2.weights_effective),
        tickers_effective=tickers_eff,
    )


@dataclass
class Level3Result:
    """Foto serializable de ``Level3_PortfolioAnalyzer`` tras ``run()``.

    Reúne todo lo que consumen el notebook 01 (vitrina principal) y la
    página de stress (que necesita ``asset_returns`` y ``weights_effective``
    para el reverse).
    """

    portfolio_name: str
    benchmark: str
    port_metrics: dict
    bench_metrics: Optional[dict]
    alpha_beta: Optional[tuple[float, float]]
    tracking_error: Optional[float]
    info_ratio: Optional[float]
    r_squared: Optional[float]
    up_down_capture: Optional[tuple[float, float]]
    kupiec: dict
    div_ratio: float
    mrc: pd.Series
    prc: pd.Series
    corr_matrix: pd.DataFrame
    anomalies: pd.Series
    rf_results: dict
    spearman: dict
    mc_results: dict
    asset_returns: pd.DataFrame
    port_returns: pd.Series
    bench_returns: Optional[pd.Series]
    weights_effective: tuple[float, ...]
    tickers_effective: tuple[str, ...]


@st.cache_data(show_spinner=False)
def analyze_level3(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    start: str,
    end: str,
    source: str,
    return_type: str = "log",
    benchmark: str = "SPY",
    portfolio_name: str = "Cartera",
    mc_sims: int = 1000,
    mc_days: int = 252,
) -> Level3Result:
    """Ejecuta ``Level3_PortfolioAnalyzer.run()`` y devuelve sus atributos."""
    ensure_data_source(source)
    lvl3 = Level3_PortfolioAnalyzer(
        tickers=list(tickers),
        weights=list(weights),
        portfolio_name=portfolio_name,
        benchmark=benchmark,
        start_date=start,
        end_date=end,
        return_type=return_type,
        mc_sims=mc_sims,
        mc_days=mc_days,
    ).run()
    return Level3Result(
        portfolio_name=portfolio_name,
        benchmark=benchmark,
        port_metrics=lvl3._port_metrics,
        bench_metrics=lvl3._bench_metrics,
        alpha_beta=(
            (float(lvl3._alpha_beta[0]), float(lvl3._alpha_beta[1]))
            if lvl3._alpha_beta is not None
            else None
        ),
        tracking_error=(
            float(lvl3._tracking_error) if lvl3._tracking_error is not None else None
        ),
        info_ratio=(
            float(lvl3._info_ratio) if lvl3._info_ratio is not None else None
        ),
        r_squared=float(lvl3._r2) if lvl3._r2 is not None else None,
        up_down_capture=(
            (float(lvl3._up_down_capture[0]), float(lvl3._up_down_capture[1]))
            if lvl3._up_down_capture is not None
            else None
        ),
        kupiec=lvl3._kupiec or {},
        div_ratio=float(lvl3._div_ratio),
        mrc=lvl3._mrc,
        prc=lvl3._prc,
        corr_matrix=lvl3._corr_matrix,
        anomalies=lvl3._anomalies,
        rf_results=lvl3._rf_results,
        spearman=lvl3._spearman,
        mc_results=lvl3._mc_results,
        asset_returns=lvl3.asset_returns,
        port_returns=lvl3.port_returns,
        bench_returns=lvl3.bench_returns,
        weights_effective=tuple(float(w) for w in lvl3.weights_effective),
        tickers_effective=tuple(lvl3.asset_returns.columns.tolist()),
    )


# ── Stress testing wrappers ─────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def run_historical_stress(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    scenario_key: str,
    source: str,
    return_type: str = "log",
    initial_value: float = 100_000.0,
) -> dict:
    """Wraps ``historical_stress_test`` con caché. Descarga la ventana en demo."""
    ensure_data_source(source)
    return historical_stress_test(
        tickers=list(tickers),
        weights=list(weights),
        scenario=scenario_key,
        return_type=return_type,
        initial_value=initial_value,
    )


@st.cache_data(show_spinner=False)
def run_historical_battery(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    source: str,
    return_type: str = "log",
    initial_value: float = 100_000.0,
) -> pd.DataFrame:
    """Wraps ``historical_stress_battery``. Ejecuta los 8 escenarios."""
    ensure_data_source(source)
    return historical_stress_battery(
        tickers=list(tickers),
        weights=list(weights),
        return_type=return_type,
        initial_value=initial_value,
    )


@st.cache_data(show_spinner=False)
def run_predefined_shock(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    shock_key: str,
    ticker_to_class: tuple[tuple[str, str], ...],
    initial_value: float = 100_000.0,
) -> dict:
    """Wraps ``apply_shock`` para shocks predefinidos. **No descarga datos.**"""
    return apply_shock(
        weights=list(weights),
        tickers=list(tickers),
        shock=shock_key,
        ticker_to_class=dict(ticker_to_class),
        initial_value=initial_value,
    )


@st.cache_data(show_spinner=False)
def run_custom_shock(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    ticker_to_class: tuple[tuple[str, str], ...],
    name: str,
    class_shocks: tuple[tuple[str, float], ...],
    ticker_overrides: tuple[tuple[str, float], ...] = (),
    default_shock: float = 0.0,
    description: str = "",
    initial_value: float = 100_000.0,
) -> dict:
    """Wraps ``apply_shock`` para un ``FactorShock`` instanciado a medida."""
    shock = FactorShock(
        name=name,
        class_shocks=dict(class_shocks),
        ticker_overrides=dict(ticker_overrides),
        default_shock=default_shock,
        description=description,
    )
    return apply_shock(
        weights=list(weights),
        tickers=list(tickers),
        shock=shock,
        ticker_to_class=dict(ticker_to_class),
        initial_value=initial_value,
    )


@st.cache_data(show_spinner=False)
def run_reverse_test(
    asset_returns: pd.DataFrame,
    weights: tuple[float, ...],
    target_loss: float,
    horizon: str = "annual",
) -> dict:
    """Wraps ``reverse_stress_test``. Recibe ``asset_returns`` ya cargados."""
    return reverse_stress_test(
        asset_returns=asset_returns,
        weights=list(weights),
        target_loss=target_loss,
        horizon=horizon,
    )


@st.cache_data(show_spinner=False)
def run_reverse_curve(
    asset_returns: pd.DataFrame,
    weights: tuple[float, ...],
    losses: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    horizon: str = "annual",
) -> pd.DataFrame:
    """Wraps ``reverse_stress_curve``."""
    return reverse_stress_curve(
        asset_returns=asset_returns,
        weights=list(weights),
        losses=list(losses),
        horizon=horizon,
    )


# ── Nivel 4 — patrimonio global + rescalado de presentación (D4) ────────────

#: Valor financiero de referencia que ``riskpkg.Level4_PatrimonyAnalyzer``
#: hardcodea internamente (ver docs/API_MAP.md §5/D5). Sirve como denominador
#: para invertir el ``+= fin_value_ref`` que hace ``riskpkg`` en
#: ``by_class['equity']`` y ``by_liquidity['liquid']``.
RISKPKG_FIN_VALUE_REF = 1_000_000.0


@dataclass
class Level4Result:
    """Foto serializable del análisis de Nivel 4 **tras rescalar** las cifras
    de agregación patrimonial al ``real_value`` indicado por el usuario.

    Las métricas financieras (``financial_metrics``) son las que devuelve
    ``riskpkg`` y NO se tocan — sólo dependen de la serie de retornos.

    Las cifras agregadas (``total_value``, ``fin_weight``,
    ``nf_items[].weight/risk_contrib``, ``var_patrimonial``,
    ``by_class``, ``by_liquidity``) se reescriben aplicando la **misma
    fórmula** que ``Level4_PatrimonyAnalyzer.run()`` pero con
    ``fin_value_ref = real_value`` en lugar del 1 M€ hardcoded.

    El campo ``original_fin_value_ref`` se conserva para que la UI pueda
    explicar el ajuste al usuario.
    """

    patrimony_name: str
    real_value: float  # valor monetario real de la cartera financiera (€)
    original_fin_value_ref: float  # 1_000_000 (lo que riskpkg fija)
    financial_metrics: dict
    nf_total_value: float
    total_value: float
    fin_weight: float
    nf_items: list[dict]
    var_patrimonial: float
    by_class: dict[str, float]
    by_liquidity: dict[str, float]


def _rescale_level4(results: dict, real_value: float) -> Level4Result:
    """Aplica el rescalado de presentación a ``Level4_PatrimonyAnalyzer._results``.

    Reproduce **bit a bit** las fórmulas internas de ``riskpkg`` con
    ``fin_value_ref = real_value``. Por construcción, si
    ``real_value == 1_000_000`` el resultado es idéntico al original.
    """
    original_ref = float(results["fin_value_ref"])
    nf_total = float(results["nf_total_value"])
    total_new = real_value + nf_total
    fin_weight_new = real_value / total_new if total_new > 0 else 1.0

    # nf_items: recalcula peso y contribución al riesgo con el nuevo total.
    new_items: list[dict] = []
    for it in results["nf_items"]:
        value = float(it["value"])
        w_new = value / total_new if total_new > 0 else 0.0
        new_items.append(
            {
                "name": it["name"],
                "class": it["class"],
                "liquidity": it["liquidity"],
                "value": value,
                "weight": w_new,
                "vol_est": float(it["vol_est"]),
                "return_est": float(it["return_est"]),
                "corr_mkt": float(it["corr_mkt"]),
                "risk_contrib": w_new * float(it["vol_est"]) * float(it["corr_mkt"]),
            }
        )

    fm = results["financial_metrics"]
    var_pat_new = fin_weight_new * float(fm["var_hist_95"]) + sum(
        it["risk_contrib"] * 1.645 for it in new_items
    )

    # Invertir el `+= fin_value_ref` que riskpkg suma al bucket
    # equity / liquid, y volver a sumar el valor real.
    by_class_new = {k: float(v) for k, v in results["by_class"].items()}
    by_class_new["equity"] = by_class_new.get("equity", 0.0) - original_ref + real_value

    by_liquidity_new = {k: float(v) for k, v in results["by_liquidity"].items()}
    by_liquidity_new["liquid"] = (
        by_liquidity_new.get("liquid", 0.0) - original_ref + real_value
    )

    return Level4Result(
        patrimony_name=results["patrimony_name"],
        real_value=real_value,
        original_fin_value_ref=original_ref,
        financial_metrics=fm,
        nf_total_value=nf_total,
        total_value=total_new,
        fin_weight=fin_weight_new,
        nf_items=new_items,
        var_patrimonial=var_pat_new,
        by_class=by_class_new,
        by_liquidity=by_liquidity_new,
    )


@st.cache_data(show_spinner=False)
def analyze_level4(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    start: str,
    end: str,
    source: str,
    benchmark: str,
    portfolio_name: str,
    mc_sims: int,
    mc_days: int,
    return_type: str,
    non_financial_assets: tuple[tuple[tuple[str, object], ...], ...],
    patrimony_name: str,
    real_value: float = RISKPKG_FIN_VALUE_REF,
) -> Level4Result:
    """Ejecuta el pipeline completo Nivel 3 → Nivel 4 y aplica rescalado D4.

    ``non_financial_assets`` se acepta como tupla-de-tuplas para que la caché
    de Streamlit pueda hashearla; cada item se reconstruye a ``dict`` justo
    antes de pasar a ``riskpkg``.
    """
    ensure_data_source(source)
    lvl3 = Level3_PortfolioAnalyzer(
        tickers=list(tickers),
        weights=list(weights),
        portfolio_name=portfolio_name,
        benchmark=benchmark,
        start_date=start,
        end_date=end,
        return_type=return_type,
        mc_sims=mc_sims,
        mc_days=mc_days,
    ).run()

    nf_list = [dict(item) for item in non_financial_assets]
    lvl4 = Level4_PatrimonyAnalyzer(
        financial_portfolio=lvl3,
        non_financial_assets=nf_list,
        patrimony_name=patrimony_name,
    ).run()

    return _rescale_level4(lvl4._results, real_value=real_value)
