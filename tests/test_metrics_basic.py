"""
Tests del núcleo de métricas: volatilidad, VaR (paramétrico e histórico),
Expected Shortfall, Sharpe y Sortino.

Oráculos cerrados siempre que la fórmula lo permite. Para métricas
muestrales (ES histórico) se valida la propiedad teórica ES ≥ VaR.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from riskpkg.metrics.basic import (
    expected_shortfall,
    sharpe_ratio,
    sortino_ratio,
    var_historical,
    var_parametric,
    volatility,
)
from riskpkg.utils.constants import TRADING_DAYS_YEAR


# ── volatility ───────────────────────────────────────────────────────────
def test_volatility_anualizada(daily_returns):
    """vol_ann == std muestral * sqrt(252)."""
    expected = daily_returns.std() * math.sqrt(TRADING_DAYS_YEAR)
    assert volatility(daily_returns) == pytest.approx(expected, rel=1e-12)


def test_volatility_diaria(daily_returns):
    assert volatility(daily_returns, annualize=False) == pytest.approx(
        daily_returns.std(), rel=1e-12
    )


# ── VaR paramétrico ──────────────────────────────────────────────────────
def test_var_parametric_formula_cerrada(daily_returns):
    """VaR_param = -(μ + z * σ) con z = Φ⁻¹(1-conf)."""
    conf = 0.95
    z = norm.ppf(1 - conf)
    expected = -(daily_returns.mean() + z * daily_returns.std())
    assert var_parametric(daily_returns, conf) == pytest.approx(expected, rel=1e-12)


def test_var_parametric_mas_estricto_a_mayor_confianza(daily_returns):
    """Confianza 99% debe producir un VaR mayor (más pérdida) que 95%."""
    assert var_parametric(daily_returns, 0.99) > var_parametric(daily_returns, 0.95)


# ── VaR histórico ────────────────────────────────────────────────────────
def test_var_historical_es_percentil_negativo(daily_returns):
    """VaR_hist al 95% = -percentil(5%) de los retornos."""
    expected = -np.percentile(daily_returns, 5)
    assert var_historical(daily_returns, 0.95) == pytest.approx(expected, rel=1e-12)


def test_var_historical_aumenta_con_confianza(daily_returns):
    assert var_historical(daily_returns, 0.99) >= var_historical(daily_returns, 0.95)


# ── Expected Shortfall ───────────────────────────────────────────────────
def test_expected_shortfall_es_al_menos_var(daily_returns):
    """Propiedad estructural: ES ≥ VaR al mismo nivel de confianza."""
    conf = 0.95
    assert expected_shortfall(daily_returns, conf) >= var_historical(daily_returns, conf)


def test_expected_shortfall_oracle_cerrado():
    """
    Con una serie controlada, ES debe ser exactamente la media de la cola.
    Construimos retornos donde el 5% peor son los 5 valores más bajos.
    """
    valores = list(np.linspace(-0.10, 0.05, 100))  # 100 retornos, ya ordenados
    r = pd.Series(valores)
    # Percentil 5 → returns[5] aprox; el percentile() interpola
    threshold = np.percentile(r, 5)
    expected = -r[r <= threshold].mean()
    assert expected_shortfall(r, 0.95) == pytest.approx(expected, rel=1e-12)


# ── Sharpe ───────────────────────────────────────────────────────────────
def test_sharpe_formula_cerrada(daily_returns):
    rf = 0.03
    excess = daily_returns.mean() * TRADING_DAYS_YEAR - rf
    vol = daily_returns.std() * math.sqrt(TRADING_DAYS_YEAR)
    expected = excess / vol
    assert sharpe_ratio(daily_returns, rf) == pytest.approx(expected, rel=1e-12)


def test_sharpe_devuelve_nan_si_vol_cero():
    """Si la serie no tiene varianza, Sharpe es indefinido."""
    r = pd.Series(np.zeros(252))
    assert math.isnan(sharpe_ratio(r, 0.0))


# ── Sortino ──────────────────────────────────────────────────────────────
def test_sortino_solo_penaliza_caidas(daily_returns):
    """
    Sortino usa solo la desviación de retornos negativos. Por construcción
    su denominador es ≤ desviación total, así que Sortino ≥ Sharpe (a igual rf).
    """
    rf = 0.0
    assert sortino_ratio(daily_returns, rf) >= sharpe_ratio(daily_returns, rf)


def test_sortino_devuelve_nan_si_sin_caidas():
    """Sin retornos negativos no hay downside → Sortino indefinido."""
    r = pd.Series(np.full(252, 0.001))   # todos positivos
    assert math.isnan(sortino_ratio(r, 0.0))
