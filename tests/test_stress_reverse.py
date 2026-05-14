"""
Tests del reverse stress test con solución cerrada de Mahalanobis.

Validamos las dos propiedades teóricas clave (Breuer & Csiszár 2013):

  1. El shock óptimo s* = -L · Σw / (w^T Σ w) produce exactamente
     una pérdida w^T s* = -L en la cartera (oráculo cerrado).

  2. La distancia de Mahalanobis del shock óptimo vale d* = L / σ_p,
     donde σ_p es la volatilidad de la cartera al horizonte indicado.
"""
import math

import numpy as np
import pytest

from riskpkg.stress.reverse import reverse_stress_curve, reverse_stress_test
from riskpkg.utils.constants import TRADING_DAYS_YEAR


# ── Propiedad 1: pérdida producida == pérdida objetivo ───────────────────
def test_reverse_loss_check_iguala_target_loss(multiasset_returns, weights3):
    """w^T s* debe ser exactamente -target_loss (con tolerancia numérica)."""
    L = 0.20
    r = reverse_stress_test(multiasset_returns, weights3, target_loss=L)
    assert r["portfolio_loss_check"] == pytest.approx(-L, rel=1e-10)


# ── Propiedad 2: distancia de Mahalanobis = L / σ_p ──────────────────────
def test_reverse_mahalanobis_es_L_sobre_sigma_p_horizonte_diario(
    multiasset_returns, weights3,
):
    L = 0.10
    cov = multiasset_returns.cov().values
    sigma_p = math.sqrt(weights3 @ cov @ weights3)

    r = reverse_stress_test(multiasset_returns, weights3, target_loss=L, horizon="daily")
    assert r["mahalanobis_distance"] == pytest.approx(L / sigma_p, rel=1e-12)
    assert r["sigma_portfolio"] == pytest.approx(sigma_p, rel=1e-12)


def test_reverse_mahalanobis_horizonte_anual_usa_covarianza_escalada(
    multiasset_returns, weights3,
):
    """En horizonte anual la covarianza se multiplica por 252."""
    L = 0.20
    cov_ann = multiasset_returns.cov().values * TRADING_DAYS_YEAR
    sigma_p_ann = math.sqrt(weights3 @ cov_ann @ weights3)

    r = reverse_stress_test(multiasset_returns, weights3, target_loss=L, horizon="annual")
    assert r["sigma_portfolio"] == pytest.approx(sigma_p_ann, rel=1e-12)
    assert r["mahalanobis_distance"] == pytest.approx(L / sigma_p_ann, rel=1e-12)


# ── Forma analítica del shock óptimo ─────────────────────────────────────
def test_reverse_shock_proporcional_a_sigma_w(multiasset_returns, weights3):
    """
    s* es paralelo a Σw. Comprobamos cosθ(s*, Σw) ≈ -1
    (negativo porque s* va contra el sentido de la cartera).
    """
    cov = multiasset_returns.cov().values
    sigma_w = cov @ weights3
    r = reverse_stress_test(multiasset_returns, weights3, target_loss=0.15)
    s_star = r["shock_vector"].values

    cos_theta = np.dot(s_star, sigma_w) / (np.linalg.norm(s_star) * np.linalg.norm(sigma_w))
    assert cos_theta == pytest.approx(-1.0, abs=1e-9)


def test_reverse_plausibility_prob_es_cola_gaussiana(multiasset_returns, weights3):
    """
    Bajo normalidad multivariante, P(pérdida ≥ L) = 1 - Φ(L/σ_p).
    Para una pérdida de 1σ → prob ≈ 0.1587 (cola superior de la normal).
    """
    cov = multiasset_returns.cov().values
    sigma_p = math.sqrt(weights3 @ cov @ weights3)
    L = sigma_p   # exactamente 1σ

    r = reverse_stress_test(multiasset_returns, weights3, target_loss=L)
    assert r["plausibility_prob"] == pytest.approx(0.1587, abs=1e-3)


# ── Curva ────────────────────────────────────────────────────────────────
def test_reverse_curve_distancia_monotona_creciente(multiasset_returns, weights3):
    """A mayor pérdida objetivo, mayor distancia de Mahalanobis (linealidad)."""
    df = reverse_stress_curve(
        multiasset_returns, weights3,
        losses=[0.05, 0.10, 0.15, 0.20, 0.25],
    )
    assert df["mahalanobis_distance"].is_monotonic_increasing


def test_reverse_curve_probabilidad_monotona_decreciente(multiasset_returns, weights3):
    """A mayor pérdida objetivo, menor probabilidad bajo la normal."""
    df = reverse_stress_curve(
        multiasset_returns, weights3,
        losses=[0.05, 0.10, 0.15, 0.20, 0.25],
    )
    assert df["plausibility_prob"].is_monotonic_decreasing


# ── Validaciones de entrada ──────────────────────────────────────────────
def test_reverse_target_loss_no_positivo_lanza_valueerror(multiasset_returns, weights3):
    with pytest.raises(ValueError, match="positiva"):
        reverse_stress_test(multiasset_returns, weights3, target_loss=0.0)
    with pytest.raises(ValueError, match="positiva"):
        reverse_stress_test(multiasset_returns, weights3, target_loss=-0.10)


def test_reverse_pesos_mal_sumados_lanza_valueerror(multiasset_returns):
    with pytest.raises(ValueError, match="sumar 1"):
        reverse_stress_test(
            multiasset_returns,
            np.array([0.5, 0.3, 0.5]),  # suma 1.3
            target_loss=0.10,
        )


def test_reverse_dimension_incompatible_lanza_valueerror(multiasset_returns):
    with pytest.raises(ValueError, match="incompatible"):
        reverse_stress_test(
            multiasset_returns,
            np.array([0.5, 0.5]),  # 2 pesos, 3 columnas
            target_loss=0.10,
        )


def test_reverse_horizon_invalido_lanza_valueerror(multiasset_returns, weights3):
    with pytest.raises(ValueError, match="horizon"):
        reverse_stress_test(
            multiasset_returns, weights3,
            target_loss=0.10, horizon="weekly",
        )
