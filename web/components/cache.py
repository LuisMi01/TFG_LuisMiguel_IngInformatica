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
from riskpkg.levels import Level1_AssetAnalyzer, Level2_FundAnalyzer

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
