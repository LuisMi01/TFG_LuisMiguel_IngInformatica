# -*- coding: utf-8 -*-
"""Web-only demo data source (offline mode).

WHY THIS EXISTS
---------------
``riskpkg``'s level analyzers (``Level1..4``) and ``historical_stress_test``
each build their *own* ``DataLoader(...).fetch()`` internally and expose **no
injection point** (verified in docs/API_MAP.md §2/§8, decision D1).
``DataLoader.load_csv`` does exist, but nothing in the orchestration layer the
web depends on ever calls it — so the "official" local-file path cannot serve
demo mode without re-implementing the level pipelines (which would violate
WEB_SPEC rule #1, single source of truth).

The only non-invasive way (respecting rule #6, "do not modify riskpkg") to make
the demo run fully offline is to intercept ``yfinance.download`` at the process
level and serve pre-fetched prices from ``web/data_cache/*.parquet``. The level
analyzers then run unchanged on cached data.

**THIS IS WEB-ONLY CODE.** It never runs inside ``riskpkg`` and never touches
the network. Generate the cache once with ``scripts/prefetch_demo.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import yfinance as yf

#: Carpeta de series congeladas para el modo demo.
DATA_CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"

#: Referencia a la función real, para poder restaurarla.
_ORIGINAL_DOWNLOAD = yf.download
_active = False


def cached_tickers() -> set[str]:
    """Tickers disponibles en el cache (por nombre de fichero parquet)."""
    if not DATA_CACHE_DIR.exists():
        return set()
    return {p.stem for p in DATA_CACHE_DIR.glob("*.parquet")}


def load_cached_close(ticker: str) -> Optional[pd.Series]:
    """Serie de cierres ajustados de un ticker, o None si no está cacheado."""
    path = DATA_CACHE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    series = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    series.name = ticker
    return series


def _fake_download(
    tickers: Union[str, Sequence[str]],
    start: Optional[str] = None,
    end: Optional[str] = None,
    **_kwargs,
) -> pd.DataFrame:
    """Sustituto offline de ``yfinance.download``.

    Devuelve un DataFrame con columnas MultiIndex ``("Close", ticker)`` para los
    tickers cacheados solicitados, recortado a ``[start, end)`` para imitar la
    semántica de yfinance (``end`` exclusivo). Los tickers no cacheados se omiten
    (``DataLoader`` los excluye con aviso y renormaliza pesos, comportamiento ya
    documentado en riskpkg).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data: dict[str, pd.Series] = {}
    for ticker in tickers:
        series = load_cached_close(ticker)
        if series is not None:
            data[ticker] = series

    if not data:
        # DataLoader lanzará un ValueError claro ("No hay datos ...").
        return pd.DataFrame()

    prices = pd.DataFrame(data)
    if start is not None:
        prices = prices[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices[prices.index < pd.Timestamp(end)]  # yfinance: end exclusivo

    # Estructura MultiIndex idéntica a yfinance ≥ 0.2 → DataLoader hace raw["Close"].
    prices.columns = pd.MultiIndex.from_tuples([("Close", t) for t in prices.columns])
    return prices


def activate() -> None:
    """Activa el modo offline: ``yfinance.download`` lee del cache local."""
    global _active
    if not _active:
        yf.download = _fake_download  # type: ignore[assignment]
        _active = True


def deactivate() -> None:
    """Restaura el ``yfinance.download`` real (modo live)."""
    global _active
    if _active:
        yf.download = _ORIGINAL_DOWNLOAD  # type: ignore[assignment]
        _active = False


def is_active() -> bool:
    return _active
