# -*- coding: utf-8 -*-
"""
Métricas comparativas frente a un benchmark.
Alpha de Jensen, Beta, Tracking Error, Information Ratio, Capture Ratios, R².
"""
from typing import Tuple

import numpy as np
import pandas as pd

from ..utils.constants import TRADING_DAYS_YEAR, DEFAULT_RISK_FREE

# Fallback gracioso si statsmodels no está disponible
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[AVISO] Librería 'statsmodels' no instalada. "
          "Alpha/Beta OLS no disponible. Instalar con: pip install statsmodels")


def alpha_beta(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    risk_free: float = DEFAULT_RISK_FREE,
) -> Tuple[float, float]:
    """
    Alpha de Jensen y Beta mediante regresión OLS.
    α = Rp - [Rf + β·(Rm - Rf)]
    """
    if HAS_STATSMODELS:
        rf_daily = risk_free / TRADING_DAYS_YEAR
        excess_p = port_returns - rf_daily
        excess_m = bench_returns - rf_daily
        X = sm.add_constant(excess_m)
        model = sm.OLS(excess_p, X).fit()
        alpha_daily = model.params.iloc[0]
        beta = model.params.iloc[1]
        return alpha_daily * TRADING_DAYS_YEAR, beta
    else:
        # Fallback sin statsmodels
        cov = port_returns.cov(bench_returns)
        var_m = bench_returns.var()
        beta = cov / var_m if var_m != 0 else np.nan
        alpha = (port_returns.mean() - bench_returns.mean() * beta) * TRADING_DAYS_YEAR
        return alpha, beta


def tracking_error(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Tracking Error anualizado: std(Rp - Rm) * sqrt(252)."""
    active = port_returns - bench_returns
    return active.std() * np.sqrt(TRADING_DAYS_YEAR)


def information_ratio(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """IR = (Rp_anual - Rm_anual) / Tracking Error."""
    te = tracking_error(port_returns, bench_returns)
    if te == 0:
        return np.nan
    active_ann = (port_returns.mean() - bench_returns.mean()) * TRADING_DAYS_YEAR
    return active_ann / te


def up_down_capture(
    port_returns: pd.Series,
    bench_returns: pd.Series,
) -> Tuple[float, float]:
    """
    Up/Down Capture Ratios.
    Up  > 100%: supera al benchmark en mercados alcistas.
    Down < 100%: cae menos que el benchmark en mercados bajistas.
    """
    up_mask = bench_returns > 0
    down_mask = bench_returns < 0

    up_capture = (port_returns[up_mask].mean() / bench_returns[up_mask].mean() * 100
                  if up_mask.sum() > 0 else np.nan)
    down_capture = (port_returns[down_mask].mean() / bench_returns[down_mask].mean() * 100
                    if down_mask.sum() > 0 else np.nan)
    return up_capture, down_capture


def r_squared(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """R² entre retornos de la cartera y el benchmark."""
    corr = np.corrcoef(port_returns, bench_returns)[0, 1]
    return corr ** 2
