# -*- coding: utf-8 -*-
"""
Agregador de métricas individuales por activo/cartera.
"""
from typing import Dict

import pandas as pd

from ..utils.constants import (
    TRADING_DAYS_YEAR,
    DEFAULT_CONFIDENCE,
    DEFAULT_RISK_FREE,
)
from .basic import (
    volatility,
    var_parametric,
    var_historical,
    expected_shortfall,
    sharpe_ratio,
    sortino_ratio,
)
from .drawdown import max_drawdown, drawdown_recovery_period


def full_report(
    returns: pd.Series,
    label: str = "Activo",
    confidence: float = DEFAULT_CONFIDENCE,
    risk_free: float = DEFAULT_RISK_FREE,
) -> Dict:
    """Calcula todas las métricas individuales en un único dict."""
    return {
        "label":             label,
        "annual_return":     returns.mean() * TRADING_DAYS_YEAR,
        "volatility_ann":    volatility(returns),
        "var_param_95":      var_parametric(returns, 0.95),
        "var_param_99":      var_parametric(returns, 0.99),
        "var_hist_95":       var_historical(returns, 0.95),
        "var_hist_99":       var_historical(returns, 0.99),
        "expected_shortfall": expected_shortfall(returns, confidence),
        "sharpe_ratio":      sharpe_ratio(returns, risk_free),
        "sortino_ratio":     sortino_ratio(returns, risk_free),
        "max_drawdown":      max_drawdown(returns),
        "recovery_days":     drawdown_recovery_period(returns),
    }
