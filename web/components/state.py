# -*- coding: utf-8 -*-
"""Session-state model for the web layer.

Holds the single ``PortfolioConfig`` the whole app reads from. Only user
configuration lives here; every analysis is delegated to ``riskpkg``.

The non-financial-asset model deliberately uses ``asset_class`` (not the
reserved word ``class``) and translates to riskpkg's exact dict keys at the
boundary via :meth:`NonFinancialAsset.to_riskpkg_dict` (see docs/API_MAP.md §5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import streamlit as st
from riskpkg.utils.constants import (
    BENCHMARK_TICKER,
    DEFAULT_RISK_FREE,
    TRADING_DAYS_YEAR,
)

#: Fixed key under which the config lives in ``st.session_state``.
STATE_KEY = "portfolio_config"

DataSource = Literal["demo", "live"]

# ── Cartera demo canónica ───────────────────────────────────────────────────
# Espejo exacto de config/default.yaml y de los notebooks (oráculo de correctitud).
DEMO_TICKERS = ["ITX.MC", "AMZN", "TTWO", "ANA", "GLD"]
DEMO_WEIGHTS = [0.30, 0.25, 0.15, 0.20, 0.10]
DEMO_START = date(2018, 1, 1)
DEMO_END = date(2024, 12, 31)
DEMO_NAME = "Cartera Demo Internacional"

# Mapeo ticker → clase de activo para stress hipotético (espejo de 06_stress_testing.py).
DEMO_TICKER_CLASSES: dict[str, str] = {
    "ITX.MC": "equity",
    "AMZN": "equity",
    "TTWO": "equity",
    "ANA": "equity",
    "GLD": "commodity",
}

#: Clases de activo válidas en riskpkg (Level4_PatrimonyAnalyzer.ASSET_CLASS_LABELS).
ASSET_CLASSES = ["equity", "fixed_income", "real_estate", "commodity", "alternative", "cash"]
#: Niveles de liquidez válidos en riskpkg.
LIQUIDITY_LEVELS = ["liquid", "semi_liquid", "illiquid"]


def default_non_financial_assets() -> list["NonFinancialAsset"]:
    """Espejo de ``non_financial_assets`` del notebook ``01_demo_completo.py``.

    Sirve como preset de partida para Nivel 4 — el usuario puede editar la
    lista directamente desde la página.
    """
    return [
        NonFinancialAsset(
            name="Inmueble Madrid",
            value=350_000,
            asset_class="real_estate",
            liquidity="illiquid",
            est_volatility=0.06,
            annual_return_est=0.03,
            corr_market=0.25,
        ),
        NonFinancialAsset(
            name="Bono Tesoro 10Y",
            value=100_000,
            asset_class="fixed_income",
            liquidity="semi_liquid",
            est_volatility=0.04,
            annual_return_est=0.035,
            corr_market=-0.20,
        ),
        NonFinancialAsset(
            name="Fondo Infraestructuras",
            value=75_000,
            asset_class="alternative",
            liquidity="illiquid",
            est_volatility=0.08,
            annual_return_est=0.05,
            corr_market=0.40,
        ),
        NonFinancialAsset(
            name="Efectivo / Depósitos",
            value=30_000,
            asset_class="cash",
            liquidity="liquid",
            est_volatility=0.0,
            annual_return_est=0.025,
            corr_market=0.0,
        ),
    ]


@dataclass
class NonFinancialAsset:
    """Activo no financiero introducido a mano (solo Nivel 4).

    Usa ``asset_class`` en vez de ``class`` (palabra reservada). El método
    :meth:`to_riskpkg_dict` traduce a las claves EXACTAS que espera
    ``Level4_PatrimonyAnalyzer`` (``class``, ``volatility_est``,
    ``correlation_with_market`` …), no las del WEB_SPEC.
    """

    name: str
    value: float
    asset_class: str  # uno de ASSET_CLASSES
    liquidity: str  # uno de LIQUIDITY_LEVELS
    est_volatility: float
    annual_return_est: float
    corr_market: float

    def to_riskpkg_dict(self) -> dict:
        """Traduce a las claves reales que consume riskpkg (ver API_MAP.md §5)."""
        return {
            "name": self.name,
            "class": self.asset_class,
            "liquidity": self.liquidity,
            "value": self.value,
            "annual_return_est": self.annual_return_est,
            "volatility_est": self.est_volatility,
            "correlation_with_market": self.corr_market,
        }


@dataclass
class PortfolioConfig:
    """Configuración única de cartera, compartida por todas las páginas."""

    tickers: list[str]
    weights: list[float]  # normalizados a suma 1 por el sidebar antes de guardarse
    start_date: date
    end_date: date
    benchmark: str = BENCHMARK_TICKER
    risk_free_rate: float = DEFAULT_RISK_FREE
    data_source: DataSource = "demo"
    return_type: str = "log"
    portfolio_name: str = DEMO_NAME
    mc_sims: int = 1000
    mc_days: int = TRADING_DAYS_YEAR
    non_financial_assets: list[NonFinancialAsset] = field(default_factory=list)

    # ── Helpers de conveniencia ──────────────────────────────────────────────
    @property
    def start_iso(self) -> str:
        return self.start_date.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end_date.isoformat()

    @property
    def ticker_to_class(self) -> dict[str, str]:
        """Clase por ticker para stress hipotético.

        Usa el mapeo demo conocido; cualquier otro ticker se asume ``equity``.
        La página de stress (Fase 2) permitirá sobrescribirlo.
        """
        return {t: DEMO_TICKER_CLASSES.get(t, "equity") for t in self.tickers}

    def non_financial_dicts(self) -> list[dict]:
        """Lista de dicts con las claves reales de riskpkg (para Level 4)."""
        return [a.to_riskpkg_dict() for a in self.non_financial_assets]


def default_config() -> PortfolioConfig:
    """Cartera demo por defecto: la web es usable nada más arrancar."""
    return PortfolioConfig(
        tickers=list(DEMO_TICKERS),
        weights=list(DEMO_WEIGHTS),
        start_date=DEMO_START,
        end_date=DEMO_END,
        portfolio_name=DEMO_NAME,
        data_source="demo",
    )


def init_state() -> None:
    """Inicializa el estado con la cartera demo si aún no existe."""
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = default_config()


def get_config() -> PortfolioConfig:
    """Devuelve la configuración actual (inicializando si hace falta)."""
    init_state()
    return st.session_state[STATE_KEY]


def set_config(config: PortfolioConfig) -> None:
    """Persiste la configuración en session_state."""
    st.session_state[STATE_KEY] = config


def has_config() -> bool:
    """True si hay una cartera válida (al menos un ticker)."""
    cfg = st.session_state.get(STATE_KEY)
    return bool(cfg and cfg.tickers)
