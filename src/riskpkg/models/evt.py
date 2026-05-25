# -*- coding: utf-8 -*-
"""
Teoría de Valores Extremos (EVT) — Peaks Over Threshold (POT).

Modela la cola de la distribución de retornos mediante una Generalized
Pareto Distribution (GPD), evitando el supuesto de normalidad que
infraestima sistemáticamente las pérdidas extremas. Permite calcular VaR
y ES a niveles de confianza arbitrariamente altos (99%, 99.5%, 99.97%)
mediante extrapolación analítica de la cola.

Resultado teórico (Pickands–Balkema–de Haan):
    Para distribuciones de retornos en el dominio de atracción de la GEV,
    los excesos sobre un umbral u suficientemente alto se distribuyen
    aproximadamente como GPD(ξ, β):

        F_u(y) = P(X − u ≤ y | X > u)  →  GPD(ξ, β)   cuando  u → x_F

    ξ > 0  → cola pesada tipo Pareto (típico en finanzas)
    ξ = 0  → cola exponencial (caso límite)
    ξ < 0  → cola corta acotada (inusual en retornos)

Fórmulas cerradas de VaR y ES bajo GPD (Smith 1987; McNeil-Frey-Embrechts 2015):

    VaR_α = u + (β/ξ) · [ ((n/N_u)·(1−α))^(−ξ) − 1 ]      si ξ ≠ 0
    ES_α  = (VaR_α + β − ξ·u) / (1 − ξ)                   si ξ < 1

Referencias:
    Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics".
    Balkema, A. & de Haan, L. (1974). "Residual Life Time at Great Age".
    Smith, R. L. (1987). "Estimating Tails of Probability Distributions".
    Hosking, J. R. M. & Wallis, J. R. (1987). "Parameter and quantile
        estimation for the generalized Pareto distribution" (PWM).
    McNeil, A. J., Frey, R. & Embrechts, P. (2015). "Quantitative Risk
        Management: Concepts, Techniques and Tools" (cap. 7).
    Coles, S. (2001). "An Introduction to Statistical Modeling of Extreme Values".

Aplicación regulatoria: el marco FRTB (Basilea III/IV) recomienda EVT-POT
para la calibración de ES a 97.5% y 99% en el Internal Models Approach.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import genpareto, goodness_of_fit

from ..metrics.basic import var_historical, var_parametric
from ..utils.constants import DEFAULT_CONFIDENCE


# ── Utilidades ────────────────────────────────────────────────────────────
def _losses(returns: pd.Series, tail: str) -> pd.Series:
    """
    Convierte una serie de retornos en pérdidas positivas según la cola.

    tail='left'  → pérdidas (retornos negados). Convención banking.
    tail='right' → ganancias extremas (cola superior).
    """
    if tail == "left":
        return -returns
    if tail == "right":
        return returns.copy()
    raise ValueError(f"tail debe ser 'left' o 'right', recibido: {tail!r}")


# ── Selección de umbral ──────────────────────────────────────────────────
def mean_excess(
    returns: pd.Series,
    thresholds: Optional[np.ndarray] = None,
    tail: str = "left",
) -> pd.DataFrame:
    """
    Calcula la función de exceso medio E[X − u | X > u] para varios umbrales.

    En una cola GPD-distribuida, el mean excess es lineal en u con pendiente
    ξ/(1−ξ) y ordenada (β − ξ·u₀)/(1−ξ). Permite la selección visual del
    umbral apropiado: se elige el u a partir del cual la curva se vuelve
    aproximadamente lineal.

    Por defecto evalúa 25 umbrales entre el percentil 75 y el 99 de las
    pérdidas. Devuelve un DataFrame con threshold, n_excesos y mean_excess.
    """
    losses = _losses(returns, tail)
    if thresholds is None:
        thresholds = np.percentile(losses, np.linspace(75, 99, 25))

    rows = []
    for u in thresholds:
        excesses = losses[losses > u] - u
        if len(excesses) >= 2:
            rows.append({
                "threshold":   float(u),
                "n_excesos":   int(len(excesses)),
                "mean_excess": float(excesses.mean()),
            })
    return pd.DataFrame(rows)


def select_threshold(
    returns: pd.Series,
    method: str = "percentile",
    value: float = 0.95,
    tail: str = "left",
) -> Tuple[float, pd.Series]:
    """
    Selecciona el umbral u y devuelve los excesos sobre él.

    method:
        'percentile' — u = percentil(value·100) de las pérdidas. Por defecto
                       value=0.95 (práctica habitual). Para mayor rigor,
                       puede customizarse en (0, 1).
        'absolute'   — u = value directamente (en unidades de pérdida).
                       Útil cuando el usuario fija el umbral tras inspeccionar
                       el mean excess plot.

    Devuelve (u, excesses) con excesses = pérdidas observadas menos u.
    """
    losses = _losses(returns, tail)

    if method == "percentile":
        if not 0 < value < 1:
            raise ValueError(f"value debe estar en (0, 1), recibido: {value}")
        u = float(np.percentile(losses, value * 100))
    elif method == "absolute":
        u = float(value)
    else:
        raise ValueError(
            f"method debe ser 'percentile' o 'absolute', recibido: {method!r}"
        )

    excesses = losses[losses > u] - u
    return u, excesses


# ── Ajuste GPD ────────────────────────────────────────────────────────────
def fit_gpd(excesses: pd.Series) -> Dict:
    """
    Ajusta una GPD por máxima verosimilitud (MLE). Fallback a Probability-
    Weighted Moments (PWM, Hosking-Wallis 1987) si MLE no converge o devuelve
    parámetros no físicos.

    Errores estándar aproximados por información de Fisher asintótica
    (McNeil-Frey-Embrechts 2015, p. 281), válidos si ξ > −½:
        Var(ξ)  ≈ (1 + ξ)² / n
        Var(β)  ≈ 2·β²·(1 + ξ) / n

    Devuelve dict con xi, beta, xi_se, beta_se, n, method.
    """
    excesses_arr = np.asarray(excesses, dtype=float)
    n = int(len(excesses_arr))

    if n < 10:
        return {
            "status": f"Insuficientes excesos ({n} < 10) para ajustar GPD.",
            "n":      n,
        }

    # ── Intento 1: MLE via scipy.stats.genpareto ─────────────────────────
    try:
        xi, _loc, beta = genpareto.fit(excesses_arr, floc=0)
        if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0):
            raise RuntimeError("MLE devolvió parámetros no físicos.")

        xi_se = float(np.sqrt((1 + xi) ** 2 / n))
        beta_se = float(np.sqrt(2 * beta ** 2 * (1 + xi) / n))

        return {
            "xi":      float(xi),
            "beta":    float(beta),
            "xi_se":   xi_se,
            "beta_se": beta_se,
            "n":       n,
            "method":  "MLE",
        }
    except Exception as e_mle:
        # ── Intento 2: PWM (Hosking-Wallis 1987) ─────────────────────────
        try:
            x = np.sort(excesses_arr)
            b0 = x.mean()
            i = np.arange(1, n + 1)
            b1 = np.sum(x * (i - 1)) / (n * (n - 1))
            xi_pwm = 2.0 - b0 / (b0 - 2 * b1)
            beta_pwm = b0 * (1 - xi_pwm)
            if not (np.isfinite(xi_pwm) and np.isfinite(beta_pwm) and beta_pwm > 0):
                raise RuntimeError("PWM devolvió parámetros no físicos.")
            return {
                "xi":      float(xi_pwm),
                "beta":    float(beta_pwm),
                "xi_se":   float("nan"),
                "beta_se": float("nan"),
                "n":       n,
                "method":  f"PWM (MLE falló: {str(e_mle)[:40]})",
            }
        except Exception as e_pwm:
            return {
                "status": (
                    f"No converge ni MLE ni PWM. MLE: {str(e_mle)[:30]}; "
                    f"PWM: {str(e_pwm)[:30]}"
                ),
                "n":      n,
            }


# ── VaR y ES por EVT-POT ─────────────────────────────────────────────────
def evt_var_es(
    returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE,
    threshold_pct: float = 0.95,
    tail: str = "left",
) -> Dict:
    """
    Pipeline VaR/ES por EVT-POT:
        1. Selección de umbral u = percentil threshold_pct de las pérdidas.
        2. Ajuste GPD por MLE sobre los excesos.
        3. Aplicación de las fórmulas cerradas (Smith 1987).

    Convención de signo: VaR y ES devueltos como pérdidas POSITIVAS, igual
    que `metrics.basic.var_historical` y `var_parametric`.

    Devuelve dict con u, n_total, n_excesos, xi, beta, xi_se, beta_se,
    method, confidence, var, es, tail.
    """
    losses = _losses(returns, tail)
    n_total = int(len(losses))
    u, excesses = select_threshold(returns, "percentile", threshold_pct, tail)

    fit = fit_gpd(excesses)
    if "status" in fit:
        return {**fit, "u": u, "n_total": n_total, "tail": tail}

    xi = fit["xi"]
    beta = fit["beta"]
    n_excesos = fit["n"]

    # Fórmula cerrada (McNeil-Frey-Embrechts 2015, ec. 7.18)
    p_tail = 1 - confidence
    ratio = n_total / n_excesos

    if abs(xi) > 1e-8:
        var_alpha = u + (beta / xi) * ((ratio * p_tail) ** (-xi) - 1)
    else:
        # Caso límite ξ → 0 (cola exponencial)
        var_alpha = u - beta * np.log(ratio * p_tail)

    if xi < 1:
        es_alpha = (var_alpha + beta - xi * u) / (1 - xi)
    else:
        # ES no acotado si ξ ≥ 1 (cola tan pesada que la media diverge)
        es_alpha = float("inf")

    return {
        "u":          u,
        "n_total":    n_total,
        "n_excesos":  n_excesos,
        "xi":         xi,
        "beta":       beta,
        "xi_se":      fit["xi_se"],
        "beta_se":    fit["beta_se"],
        "method":     fit["method"],
        "confidence": confidence,
        "var":        float(var_alpha),
        "es":         float(es_alpha),
        "tail":       tail,
    }


# ── Informe completo con diagnósticos ────────────────────────────────────
def evt_report(
    returns: pd.Series,
    threshold_pct: float = 0.95,
    confidences: Optional[List[float]] = None,
    tail: str = "left",
    ad_mc_samples: int = 200,
) -> Dict:
    """
    Informe EVT-POT completo con diagnóstico de bondad de ajuste.

    Test Anderson-Darling sobre la GPD ajustada vía
    `scipy.stats.goodness_of_fit` con p-valor estimado por Monte Carlo
    (necesario porque los parámetros se estiman de los mismos datos).
    AD es más sensible a las colas que Kolmogorov-Smirnov, lo cual es
    deseable en EVT — precisamente lo que importa es la cola.

    Devuelve dict con u, threshold_pct, xi, beta, errores estándar,
    var_es por nivel de confianza, ad_statistic, ad_pvalue,
    interpretation textual y método de ajuste.
    """
    if confidences is None:
        confidences = [0.95, 0.99, 0.995]

    losses = _losses(returns, tail)
    n_total = int(len(losses))
    u, excesses = select_threshold(returns, "percentile", threshold_pct, tail)

    fit = fit_gpd(excesses)
    if "status" in fit:
        return {**fit, "u": u, "n_total": n_total, "tail": tail}

    xi = fit["xi"]
    beta = fit["beta"]

    # VaR y ES a varias confianzas
    var_es = {}
    for c in confidences:
        r = evt_var_es(returns, confidence=c, threshold_pct=threshold_pct, tail=tail)
        var_es[c] = {"var": r["var"], "es": r["es"]}

    # ── Bondad de ajuste: Anderson-Darling vía goodness_of_fit ────────────
    excesses_arr = np.asarray(excesses, dtype=float)
    try:
        gof = goodness_of_fit(
            genpareto,
            excesses_arr,
            known_params={"loc": 0},
            statistic="ad",
            n_mc_samples=ad_mc_samples,
        )
        ad_stat = float(gof.statistic)
        ad_pvalue = float(gof.pvalue)
    except Exception:
        ad_stat = float("nan")
        ad_pvalue = float("nan")

    return {
        "u":              u,
        "threshold_pct":  threshold_pct,
        "tail":           tail,
        "n_total":        n_total,
        "n_excesos":      fit["n"],
        "xi":             xi,
        "beta":           beta,
        "xi_se":          fit["xi_se"],
        "beta_se":        fit["beta_se"],
        "method":         fit["method"],
        "var_es":         var_es,
        "ad_statistic":   ad_stat,
        "ad_pvalue":      ad_pvalue,
        "interpretation": _interpret_evt(xi, ad_pvalue),
    }


def _interpret_evt(xi: float, ad_pvalue: float) -> str:
    """Construye una etiqueta textual del resultado del ajuste EVT."""
    if not np.isfinite(ad_pvalue):
        gof_note = "(sin p-valor)"
    elif ad_pvalue < 0.05:
        gof_note = f"⚠️  Ajuste GPD rechazado por Anderson-Darling (p = {ad_pvalue:.3f})"
    else:
        gof_note = f"✅  Ajuste GPD aceptable (Anderson-Darling p = {ad_pvalue:.3f})"

    if xi > 0.5:
        tail_note = f"cola muy pesada (ξ = {xi:.3f} > 0.5)"
    elif xi > 0:
        tail_note = f"cola pesada Pareto (ξ = {xi:.3f})"
    elif abs(xi) < 1e-3:
        tail_note = "cola exponencial (ξ ≈ 0)"
    else:
        tail_note = f"cola corta (ξ = {xi:.3f} < 0, inusual en finanzas)"

    return f"{gof_note}. Diagnóstico: {tail_note}."


# ── Comparación con métodos clásicos ─────────────────────────────────────
def compare_var_methods(
    returns: pd.Series,
    confidence: float = 0.99,
    threshold_pct: float = 0.95,
) -> Dict:
    """
    Compara el VaR a una misma confianza usando tres métodos:

        1. Paramétrico gaussiano (asume normalidad — infraestima cola).
        2. Histórico (no paramétrico — limitado por el peor dato observado).
        3. EVT-POT (extrapolación analítica de la cola).

    Sirve como evidencia del beneficio de EVT frente a métodos clásicos:
    si VaR_EVT / VaR_paramétrico > 1.2 la cola es claramente más pesada
    que una gaussiana y el método paramétrico subestima el riesgo de cola.

    Devuelve dict con los tres VaR, el ES de EVT, el ratio EVT/paramétrico,
    el índice de cola ξ y una interpretación textual.
    """
    var_p = float(var_parametric(returns, confidence))
    var_h = float(var_historical(returns, confidence))
    evt = evt_var_es(returns, confidence=confidence, threshold_pct=threshold_pct)

    if "status" in evt:
        return {
            "confidence":     confidence,
            "var_parametric": var_p,
            "var_historical": var_h,
            "evt_status":     evt["status"],
        }

    var_e = float(evt["var"])
    es_e = float(evt["es"])
    ratio = var_e / var_p if var_p != 0 else float("nan")

    if not np.isfinite(ratio):
        interpretation = "VaR paramétrico ≈ 0; ratio indefinido."
    elif ratio > 1.2:
        interpretation = (
            f"VaR_EVT/VaR_param = {ratio:.2f}× → cola significativamente más "
            f"pesada que gaussiana. El método paramétrico subestima el riesgo de cola."
        )
    elif ratio < 0.85:
        interpretation = (
            f"VaR_EVT/VaR_param = {ratio:.2f}× → cola más ligera que gaussiana "
            f"(inusual; verificar umbral)."
        )
    else:
        interpretation = (
            f"VaR_EVT/VaR_param = {ratio:.2f}× → cola próxima a gaussiana."
        )

    return {
        "confidence":       confidence,
        "var_parametric":   var_p,
        "var_historical":   var_h,
        "var_evt":          var_e,
        "es_evt":           es_e,
        "ratio_evt_param":  ratio,
        "xi":               float(evt["xi"]),
        "interpretation":   interpretation,
    }
