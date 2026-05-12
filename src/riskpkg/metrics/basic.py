# -*- coding: utf-8 -*-
"""
Métricas básicas de riesgo y rendimiento ajustado al riesgo.

Funciones puras (sin estado). Reciben Series de retornos ya limpios y
calculan la métrica solicitada. Núcleo compartido por los cuatro niveles.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.constants import (
    TRADING_DAYS_YEAR,
    DEFAULT_CONFIDENCE,
    DEFAULT_RISK_FREE,
)


# ── Volatilidad ──────────────────────────────────────────────────────────
def volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Volatilidad histórica. Anualizada por defecto."""
    vol = returns.std()
    return vol * np.sqrt(TRADING_DAYS_YEAR) if annualize else vol


# ── VaR ──────────────────────────────────────────────────────────────────
def var_parametric(returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """VaR paramétrico gaussiano. Asume distribución normal."""
    z = norm.ppf(1 - confidence)
    return -(returns.mean() + z * returns.std())


def var_historical(returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """VaR histórico (no paramétrico). Robusto ante colas gruesas."""
    return -np.percentile(returns, (1 - confidence) * 100)


# ── Expected Shortfall (CVaR) ────────────────────────────────────────────
def expected_shortfall(returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """
    ES / CVaR: pérdida media más allá del umbral VaR.
    Medida coherente de riesgo (Artzner et al., 1999).
    Adoptada por Basilea IV (2019) como métrica principal de capital.
    """
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    return -tail.mean() if not tail.empty else 0.0


# ── Ratios de rendimiento ajustado al riesgo ─────────────────────────────
def sharpe_ratio(returns: pd.Series, risk_free: float = DEFAULT_RISK_FREE) -> float:
    """Ratio de Sharpe anualizado (Sharpe, 1994)."""
    excess = returns.mean() * TRADING_DAYS_YEAR - risk_free
    vol = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    return excess / vol if vol != 0 else np.nan


def sortino_ratio(returns: pd.Series, risk_free: float = DEFAULT_RISK_FREE) -> float:
    """
    Ratio de Sortino: penaliza solo la volatilidad a la baja.
    Más adecuado cuando la distribución de retornos es asimétrica.
    (Sortino & Price, 1994)
    """
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS_YEAR)
    excess = returns.mean() * TRADING_DAYS_YEAR - risk_free
    return excess / downside if downside != 0 else np.nan
