# -*- coding: utf-8 -*-
"""Mapeo ticker → nombre comercial para presentación en la UI.

Política
--------
- **Modo demo** (offline): sólo se usa ``DEMO_NAMES``. Nunca llamamos
  a yfinance, porque el demo es bit a bit determinista y no debe
  depender de la red.
- **Modo live**: si el ticker no está en ``DEMO_NAMES`` se consulta
  ``yfinance.Ticker(t).info['longName']``. El resultado se cachea con
  ``@st.cache_data`` para reutilizarlo entre reruns. Si falla (sin red,
  ticker raro) el helper devuelve silenciosamente el propio ticker —
  no rompe la UI.

Esta capa **no toca** ``cfg.tickers`` ni nada que viaje a ``riskpkg``:
los analizadores siguen recibiendo tickers crudos. Aquí sólo se traduce
de cara al usuario.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

#: Tickers del cache demo (web/data_cache/) y un grupo de blue chips
#: habituales para que un usuario que pruebe modo live en aula tenga
#: nombres legibles incluso si Yahoo Finance está lento o caído.
DEMO_NAMES: dict[str, str] = {
    # Cache demo del TFG
    "ITX.MC": "Inditex",
    "AMZN": "Amazon.com Inc.",
    "TTWO": "Take-Two Interactive",
    "ANA": "Acciona",
    "GLD": "SPDR Gold Shares",
    "SPY": "SPDR S&P 500 ETF",
    # Mega caps típicas en pruebas live
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "BRK-B": "Berkshire Hathaway",
}


@st.cache_data(show_spinner=False)
def _lookup_live(ticker: str) -> str | None:
    """Consulta yfinance.Ticker(t).info['longName']; ``None`` si falla."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName")
        return name if name else None
    except Exception:  # noqa: BLE001 — yfinance lanza varios tipos
        return None


def display_name(ticker: str, *, source: str) -> str:
    """Devuelve ``"Nombre (TICKER)"`` si se conoce, o ``"TICKER"`` a secas.

    Parameters
    ----------
    ticker
        Ticker crudo (p.ej. ``"AAPL"``).
    source
        ``"demo"`` o ``"live"``. En demo nunca se hace lookup live.
    """
    if not ticker:
        return ticker
    name = DEMO_NAMES.get(ticker)
    if name is None and source == "live":
        name = _lookup_live(ticker)
    return f"{name} ({ticker})" if name else ticker


def display_many(tickers: Iterable[str], *, source: str) -> list[str]:
    """Vectorización de :func:`display_name` para listas."""
    return [display_name(t, source=source) for t in tickers]


def rename_index(obj, *, source: str):
    """Devuelve una copia de ``obj`` con el índice renombrado a display name.

    Funciona con ``pd.Series`` y ``pd.DataFrame``.
    """
    return obj.rename(index=lambda t: display_name(t, source=source))


def rename_columns(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Devuelve una copia de ``df`` con las columnas renombradas."""
    return df.rename(columns=lambda t: display_name(t, source=source))


def rename_axes(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Renombra índice y columnas a display name (para matrices de correlación)."""
    fn = lambda t: display_name(t, source=source)  # noqa: E731
    return df.rename(index=fn, columns=fn)
