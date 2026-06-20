# -*- coding: utf-8 -*-
"""Disponibilidad de series temporales por ticker.

Se usa para que el usuario sepa cuánta historia hay disponible antes
de ajustar el rango de fechas en la barra lateral. Política:

- **Demo**: leemos el rango directamente del parquet local
  (``web/data_cache/<TICKER>.parquet``). Gratis, offline.
- **Live**: ``yfinance.Ticker(t).history(period="max")`` cacheado por
  sesión con ``@st.cache_data`` para no machacar la API. Si yfinance
  falla, devolvemos ``None`` — la UI lo mostrará como "—".

Esta capa NO toca ``riskpkg`` ni introduce cálculo financiero. Sólo
expone metadatos (primera fecha, última fecha, número de sesiones).
"""
from __future__ import annotations

from typing import Optional, TypedDict

import pandas as pd
import streamlit as st

from . import demo_data


class Availability(TypedDict):
    first: pd.Timestamp
    last: pd.Timestamp
    n_sessions: int


def _from_parquet(ticker: str) -> Optional[Availability]:
    series = demo_data.load_cached_close(ticker)
    if series is None or series.empty:
        return None
    return Availability(
        first=series.index.min(),
        last=series.index.max(),
        n_sessions=int(len(series)),
    )


@st.cache_data(show_spinner=False)
def _from_yfinance(ticker: str) -> Optional[Availability]:
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return Availability(
            first=hist.index.min(),
            last=hist.index.max(),
            n_sessions=int(len(hist)),
        )
    except Exception:  # noqa: BLE001 — yfinance puede lanzar varios tipos
        return None


def get_availability(ticker: str, *, source: str) -> Optional[Availability]:
    """Devuelve la ventana disponible para ``ticker`` según el modo activo."""
    if not ticker:
        return None
    if source == "demo":
        return _from_parquet(ticker)
    return _from_yfinance(ticker)


def clear_live_cache() -> None:
    """Invalida el cache de yfinance para forzar una nueva consulta."""
    _from_yfinance.clear()
