# -*- coding: utf-8 -*-
"""
Capa de acceso a datos — DataLoader.

Responsabilidad: obtener, validar y preparar datos financieros desde
fuentes externas (yfinance) o ficheros locales (CSV). Calcula retornos
simples y logarítmicos. Abstrae el origen de los datos del resto del sistema.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..utils.constants import (
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    MIN_OBSERVATIONS,
)


class DataLoader:
    """
    Responsabilidad: obtener, validar y preparar datos financieros.

    Abstrae el origen de los datos (yfinance API, CSV local) del resto
    del sistema. Calcula retornos simples y logarítmicos. Verifica que
    existan mínimo MIN_OBSERVATIONS sesiones antes de permitir el análisis.

    Uso:
        loader = DataLoader(["AAPL", "MSFT"], "2015-01-01", "2024-12-31")
        loader.fetch()
        returns = loader.log_returns   # retornos logarítmicos
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        return_type: str = "log",        # "log" | "simple"
    ):
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.return_type = return_type
        self._prices: Optional[pd.DataFrame] = None
        self._log_returns: Optional[pd.DataFrame] = None
        self._simp_returns: Optional[pd.DataFrame] = None

    # ── Obtención ─────────────────────────────────────────────────────────
    def fetch(self) -> "DataLoader":
        """Descarga precios de cierre ajustados desde yfinance."""
        print(f"[DataLoader] Descargando {self.tickers} "
              f"({self.start_date} → {self.end_date}) ...")
        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
        )
        # Compatibilidad con MultiIndex de yfinance ≥ 0.2
        if isinstance(raw.columns, pd.MultiIndex):
            self._prices = raw["Close"]
        else:
            self._prices = raw[["Close"]]
            self._prices.columns = self.tickers

        self._validate()
        return self

    def load_csv(self, path: str) -> "DataLoader":
        """Carga precios desde un CSV local (columna Date como índice)."""
        self._prices = pd.read_csv(path, index_col=0, parse_dates=True)
        self._validate()
        return self

    # ── Validación ────────────────────────────────────────────────────────
    def _validate(self) -> None:
        """Limpieza de datos: elimina columnas vacías, rellena huecos,
        verifica mínimo de observaciones."""
        if self._prices is None or self._prices.empty:
            raise ValueError("[DataLoader] No hay datos. Verifica tickers o conexión.")

        # Eliminar tickers sin datos
        empty = self._prices.columns[self._prices.isna().all()]
        if not empty.empty:
            print(f"[DataLoader] AVISO: sin datos para {list(empty)}. Excluidos.")
            self._prices.drop(columns=empty, inplace=True)

        # Forward-fill huecos menores (festivos, suspensiones cortas)
        self._prices.ffill(inplace=True)
        self._prices.dropna(inplace=True)

        # Verificación de mínimo de observaciones
        n = len(self._prices)
        if n < MIN_OBSERVATIONS:
            print(f"[DataLoader] AVISO: solo {n} observaciones "
                  f"(mínimo recomendado: {MIN_OBSERVATIONS}). "
                  "Las métricas pueden no ser fiables.")

    # ── Propiedades públicas ──────────────────────────────────────────────
    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            raise RuntimeError("Ejecuta fetch() o load_csv() primero.")
        return self._prices

    @property
    def log_returns(self) -> pd.DataFrame:
        """Retornos logarítmicos diarios: ln(P_t / P_{t-1})"""
        if self._log_returns is None:
            self._log_returns = np.log(self._prices / self._prices.shift(1)).dropna()
        return self._log_returns

    @property
    def simple_returns(self) -> pd.DataFrame:
        """Retornos simples diarios: (P_t - P_{t-1}) / P_{t-1}"""
        if self._simp_returns is None:
            self._simp_returns = self._prices.pct_change().dropna()
        return self._simp_returns

    @property
    def returns(self) -> pd.DataFrame:
        """Retorna log_returns o simple_returns según return_type configurado."""
        return self.log_returns if self.return_type == "log" else self.simple_returns

    def summary(self) -> None:
        p = self.prices
        print(f"\n{'─'*55}")
        print(f"[DataLoader] Resumen de datos cargados")
        print(f"  Tickers      : {list(p.columns)}")
        print(f"  Período      : {p.index[0].date()} → {p.index[-1].date()}")
        print(f"  Sesiones     : {len(p)}")
        print(f"  NaN restantes: {p.isna().sum().sum()}")
        print(f"  Tipo retornos: {self.return_type}")
        print(f"{'─'*55}\n")
