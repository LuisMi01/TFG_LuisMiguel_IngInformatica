# -*- coding: utf-8 -*-
"""Cached wrappers over riskpkg data access + demo-source activation.

``@st.cache_data`` is safe here because riskpkg is deterministic
(``RANDOM_SEED = 42`` is fixed on import; Monte Carlo / Isolation Forest reseed
internally). Cache keys include ``data_source`` so demo and live never collide.

Phase 0 provides price/return loaders (enough to prove offline demo mode works);
the cached level-analyzer wrappers are added in Phase 1.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from riskpkg.data import DataLoader

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
