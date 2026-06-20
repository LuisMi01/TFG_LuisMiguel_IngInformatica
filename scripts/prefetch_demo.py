# -*- coding: utf-8 -*-
"""Pre-descarga las series de la cartera demo a web/data_cache/*.parquet.

Se ejecuta UNA vez (con conexión). Congela los precios de cierre ajustados para
que el **modo demo** de la web funcione sin red y de forma reproducible.

El rango cubre desde 2000 para que el stress testing histórico (ventanas dot-com,
GFC, etc.) también funcione offline. Los activos sin datos en fechas antiguas
(p. ej. GLD desde 2004) se guardan con el histórico disponible; riskpkg los
excluye y renormaliza pesos donde falten (comportamiento ya documentado).

Uso:
    .venv-web/bin/python scripts/prefetch_demo.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

# Universo demo (espejo de config/default.yaml) + benchmark.
# Incluye ANA (cartera canónica del TFG) y AAPL (default de la web).
DEMO_TICKERS = ["ITX.MC", "AMZN", "TTWO", "ANA", "AAPL", "GLD", "SPY"]
START = "2000-01-01"
END = "2025-12-31"

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "data_cache"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[prefetch] Descargando {DEMO_TICKERS} ({START} → {END}) ...")

    raw = yf.download(DEMO_TICKERS, start=START, end=END, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    saved, missing = [], []
    for ticker in DEMO_TICKERS:
        if ticker in close.columns:
            series = close[ticker].dropna()
            if series.empty:
                missing.append(ticker)
                continue
            series.to_frame("Close").to_parquet(OUT_DIR / f"{ticker}.parquet")
            saved.append(f"{ticker} ({len(series)} sesiones, {series.index[0].date()}→{series.index[-1].date()})")
        else:
            missing.append(ticker)

    print(f"[prefetch] Guardados en {OUT_DIR}:")
    for line in saved:
        print(f"  ✅ {line}")
    if missing:
        print(f"  ⚠️ Sin datos: {missing}")


if __name__ == "__main__":
    main()
