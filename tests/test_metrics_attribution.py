"""
Tests de atribución y diversificación.

Propiedad clave validada: la suma de las contribuciones marginales al riesgo
(MRC) coincide con la volatilidad de la cartera (identidad de Euler).
"""
import math

import numpy as np
import pandas as pd
import pytest

from riskpkg.metrics.attribution import (
    diversification_benefit,
    diversification_ratio,
    marginal_risk_contribution,
    percentage_risk_contribution,
)
from riskpkg.utils.constants import TRADING_DAYS_YEAR


# ── Diversification Ratio ────────────────────────────────────────────────
def test_diversification_ratio_un_solo_activo_es_uno():
    """Con un solo activo, DR = σ/σ = 1 (sin diversificación)."""
    r = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 252)})
    dr = diversification_ratio(r, np.array([1.0]))
    assert dr == pytest.approx(1.0, abs=1e-9)


def test_diversification_ratio_mayor_que_uno_con_activos_no_correlacionados():
    """Con activos correlacionados imperfectamente, DR debe ser > 1."""
    rng = np.random.default_rng(42)
    r = pd.DataFrame(rng.normal(0, 0.01, (252, 3)), columns=["A", "B", "C"])
    dr = diversification_ratio(r, np.array([1 / 3, 1 / 3, 1 / 3]))
    assert dr > 1.0


def test_diversification_benefit_no_negativo(multiasset_returns, weights3):
    """La combinación de activos correlacionados <1 no puede aumentar la vol."""
    assert diversification_benefit(multiasset_returns, weights3) >= 0


# ── Marginal Risk Contribution ───────────────────────────────────────────
def test_mrc_devuelve_serie_indexada_por_ticker(multiasset_returns, weights3):
    mrc = marginal_risk_contribution(multiasset_returns, weights3)
    assert isinstance(mrc, pd.Series)
    assert list(mrc.index) == list(multiasset_returns.columns)


def test_mrc_suma_igual_a_volatilidad_de_cartera(multiasset_returns, weights3):
    """
    Identidad de Euler: Σ MRC_i = σ_portfolio.

    Es la propiedad estructural fundamental de la atribución de riesgo:
    el riesgo total se descompone exactamente en las contribuciones
    marginales ponderadas. Tolerancia numérica estricta (1e-10).
    """
    cov = multiasset_returns.cov().values * TRADING_DAYS_YEAR
    port_vol = math.sqrt(weights3 @ cov @ weights3)
    mrc = marginal_risk_contribution(multiasset_returns, weights3)
    assert mrc.sum() == pytest.approx(port_vol, rel=1e-10)


# ── Percentage Risk Contribution ─────────────────────────────────────────
def test_prc_suma_aproximadamente_cien(multiasset_returns, weights3):
    prc = percentage_risk_contribution(multiasset_returns, weights3)
    assert prc.sum() == pytest.approx(100.0, abs=1e-6)


def test_prc_todos_componentes_no_negativos(multiasset_returns, weights3):
    """Con pesos positivos y covarianza PSD, PRC_i ≥ 0 para todo i."""
    prc = percentage_risk_contribution(multiasset_returns, weights3)
    assert (prc >= 0).all()
