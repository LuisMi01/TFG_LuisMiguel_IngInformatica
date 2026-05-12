# -*- coding: utf-8 -*-
"""
Reverse stress testing: dada una pérdida objetivo, identificar el escenario
de mercado más plausible (en sentido estadístico) que la generaría.

Metodología — Breuer & Csiszár (2013):
    "Systematic stress tests with entropic plausibility constraints"

Formulación matemática
    Sean:
        w ∈ R^n     vector de pesos de la cartera
        Σ ∈ R^{nxn} matriz de covarianzas de retornos
        L > 0       pérdida objetivo (en proporción del valor de cartera)

    Problema:
        min  s^T Σ^(-1) s          (distancia de Mahalanobis al cuadrado)
        s.a. w^T s = -L            (la cartera pierde exactamente L)

    Solución (Lagrange):
        s* = -L · (Σ w) / (w^T Σ w)
        d_Mahalanobis(s*) = L / σ_p

    Interpretación:
        El shock óptimo es paralelo a Σw, es decir, redistribuye la pérdida
        proporcionalmente a la contribución marginal al riesgo de cada activo.
        Su distancia de Mahalanobis equivale al número de desviaciones típicas
        de la cartera que representa la pérdida — equivale al z-score bajo
        normalidad multivariante.

Aplicación bancaria
    Este test responde a preguntas como:
        "¿Qué combinación de movimientos de mercado más cercana al
        comportamiento normal causaría una pérdida del 20% de la cartera?"

    Es el inverso conceptual del VaR: en lugar de "dada una probabilidad,
    qué pérdida esperamos", se pregunta "dada una pérdida, qué escenario
    plausible la produce".
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.constants import TRADING_DAYS_YEAR


def reverse_stress_test(
    asset_returns: pd.DataFrame,
    weights: Union[List[float], np.ndarray],
    target_loss: float,
    horizon: str = "daily",   # "daily" | "annual"
) -> Dict:
    """
    Calcula el shock más plausible (mínima distancia de Mahalanobis) que
    causaría una pérdida igual a `target_loss` en la cartera.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Retornos diarios de los activos (columnas = tickers).
    weights : array-like
        Pesos de cartera correspondientes a las columnas (suman 1).
    target_loss : float
        Pérdida objetivo en proporción (e.g., 0.20 = 20% de pérdida).
        Debe ser estrictamente positiva.
    horizon : str
        "daily" — el shock es de horizonte 1 día (covarianza no anualizada).
        "annual" — el shock es de horizonte anual (covarianza × 252).

    Returns
    -------
    dict con:
        - target_loss              : pérdida objetivo solicitada
        - horizon                  : horizonte del shock
        - shock_vector             : pd.Series indexada por ticker con el shock
                                     óptimo (negativo = caída del activo)
        - portfolio_loss_check     : pérdida que produce el shock (verificación)
        - mahalanobis_distance     : distancia del escenario al "normal" (en σ)
        - plausibility_zscore      : igual que la distancia (interpretación gauss.)
        - plausibility_prob        : probabilidad de un escenario al menos tan
                                     extremo bajo normalidad multivariante
        - sigma_portfolio          : volatilidad de cartera al horizonte indicado
    """
    if target_loss <= 0:
        raise ValueError("target_loss debe ser estrictamente positiva.")

    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Los pesos deben sumar 1. Suma actual: {weights.sum():.6f}")

    if asset_returns.shape[1] != len(weights):
        raise ValueError(
            f"Dimensión incompatible: {asset_returns.shape[1]} activos vs "
            f"{len(weights)} pesos."
        )

    # Matriz de covarianzas según horizonte
    cov_daily = asset_returns.cov().values
    if horizon == "annual":
        cov = cov_daily * TRADING_DAYS_YEAR
    elif horizon == "daily":
        cov = cov_daily
    else:
        raise ValueError(f"horizon debe ser 'daily' o 'annual', recibido: {horizon}")

    # Volatilidad de cartera al horizonte
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        raise RuntimeError(
            "Varianza de cartera no positiva. Matriz de covarianzas singular."
        )
    sigma_p = float(np.sqrt(port_var))

    # ── Solución cerrada: s* = -L · Σw / (w^T Σ w) ──────────────────────────
    sigma_w = cov @ weights
    shock_optimal = -target_loss * sigma_w / port_var

    # Verificación: w^T s* debe ser -target_loss
    loss_check = float(weights @ shock_optimal)

    # ── Plausibilidad (distancia de Mahalanobis) ────────────────────────────
    # d² = s* Σ^(-1) s* = target_loss² / σ_p²
    # Equivale al número de desviaciones típicas que representa la pérdida.
    mahalanobis = target_loss / sigma_p

    # Probabilidad cola izquierda bajo normalidad multivariante
    # P(pérdida ≥ target_loss) = 1 - Φ(target_loss / σ_p)
    prob_tail = float(1 - norm.cdf(mahalanobis))

    shock_series = pd.Series(
        shock_optimal, index=asset_returns.columns, name="shock_optimal"
    )

    return {
        "target_loss":             target_loss,
        "horizon":                 horizon,
        "shock_vector":            shock_series,
        "portfolio_loss_check":    loss_check,
        "mahalanobis_distance":    mahalanobis,
        "plausibility_zscore":     mahalanobis,
        "plausibility_prob":       prob_tail,
        "sigma_portfolio":         sigma_p,
    }


def reverse_stress_curve(
    asset_returns: pd.DataFrame,
    weights: Union[List[float], np.ndarray],
    losses: Optional[List[float]] = None,
    horizon: str = "daily",
) -> pd.DataFrame:
    """
    Genera la curva de plausibilidad pérdida → distancia de Mahalanobis.
    Útil para gráficas y para identificar el "punto de inflexión" donde
    la pérdida deja de ser plausible.

    Parameters
    ----------
    losses : List[float], opcional
        Niveles de pérdida a evaluar. Por defecto: [5%, 10%, 15%, ..., 50%]
    """
    if losses is None:
        losses = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    rows = []
    for L in losses:
        r = reverse_stress_test(asset_returns, weights, L, horizon=horizon)
        rows.append({
            "target_loss_pct":      L,
            "mahalanobis_distance": r["mahalanobis_distance"],
            "plausibility_prob":    r["plausibility_prob"],
            "sigma_portfolio":      r["sigma_portfolio"],
        })

    return pd.DataFrame(rows).set_index("target_loss_pct")
