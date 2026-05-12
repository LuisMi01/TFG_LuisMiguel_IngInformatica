# -*- coding: utf-8 -*-
"""
Métricas de atribución y diversificación del riesgo de cartera.
Ratio de diversificación, beneficio de diversificación,
contribución marginal y porcentual al riesgo.
"""
import numpy as np
import pandas as pd

from ..utils.constants import TRADING_DAYS_YEAR


def diversification_ratio(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Ratio de Diversificación (Choueifaty & Coignard, 2008).
    DR = (Σ wᵢ·σᵢ) / σ_portfolio
    DR > 1 indica beneficio de diversificación.
    """
    individual_vols = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    cov_matrix = returns.cov() * TRADING_DAYS_YEAR
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    weighted_vols = np.dot(weights, individual_vols.values)
    return weighted_vols / port_vol if port_vol != 0 else np.nan


def diversification_benefit(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Beneficio de Diversificación (en puntos porcentuales anualizados).
    = Σ wᵢ·σᵢ - σ_portfolio
    Cuantifica la reducción de riesgo obtenida por la combinación de activos.
    """
    individual_vols = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
    cov_matrix = returns.cov() * TRADING_DAYS_YEAR
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    weighted_vols = np.dot(weights, individual_vols.values)
    return weighted_vols - port_vol


def marginal_risk_contribution(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Contribución Marginal al Riesgo (MRC) por componente.
    MCRᵢ = wᵢ · (Σw)ᵢ / σ_portfolio
    Suma total = volatilidad de la cartera.

    Permite identificar qué activos aportan más riesgo relativo.
    """
    cov_matrix = returns.cov() * TRADING_DAYS_YEAR
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    mrc = weights * np.dot(cov_matrix, weights) / port_vol
    return pd.Series(mrc, index=returns.columns, name="MRC")


def percentage_risk_contribution(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Contribución Porcentual al Riesgo (% del riesgo total de la cartera).
    PRC_i = MRC_i / σ_portfolio
    """
    mrc = marginal_risk_contribution(returns, weights)
    total = mrc.sum()
    return (mrc / total * 100).rename("PRC (%)")
