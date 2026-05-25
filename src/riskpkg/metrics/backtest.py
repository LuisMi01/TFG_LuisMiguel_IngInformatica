# -*- coding: utf-8 -*-
"""
Backtesting de modelos VaR. Test de Kupiec (1995).
"""
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..utils.constants import DEFAULT_CONFIDENCE
from .basic import var_historical, var_parametric


def kupiec_test(
    returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE,
    var_method: str = "historical",    # "historical" | "parametric"
) -> Dict:
    """
    Test de Kupiec (1995) para el backtesting del VaR.
    Contrasta si la tasa de excedencias empírica coincide con la teórica.

    H0: La proporción de excepciones es igual al nivel esperado (1-conf).
    Estadístico: razón de verosimilitud LR ~ chi2(1) bajo H0.

    Requiere mínimo 250 observaciones out-of-sample.

    Retorna dict con: n, exceedances, p_expected, p_actual, lr_stat, p_value,
                     reject_h0 (True = modelo mal especificado al 5%).
    """
    n = len(returns)
    if n < 250:
        return {"status": "Insuficientes observaciones (< 250) para test de Kupiec."}

    # Calcular VaR para cada día usando los datos previos (walk-forward)
    window = min(252, n // 2)
    exceedances = 0
    valid_days = 0

    for i in range(window, n):
        hist = returns.iloc[i - window:i]
        if var_method == "historical":
            var_t = var_historical(hist, confidence)
        else:
            var_t = var_parametric(hist, confidence)

        actual_return = returns.iloc[i]
        if actual_return < -var_t:
            exceedances += 1
        valid_days += 1

    p_expected = 1 - confidence
    p_actual = exceedances / valid_days if valid_days > 0 else 0

    # Estadístico LR de Kupiec
    if p_actual == 0:
        lr_stat = 2 * valid_days * np.log((1 - p_expected))
    elif p_actual == 1:
        lr_stat = 2 * valid_days * np.log(p_expected)
    else:
        lr_stat = (
            2 * (exceedances * np.log(p_actual / p_expected) +
                 (valid_days - exceedances) * np.log((1 - p_actual) / (1 - p_expected)))
        )

    p_value = 1 - chi2.cdf(lr_stat, df=1)
    reject_h0 = p_value < 0.05

    return {
        "n_total":        n,
        "n_test":         valid_days,
        "exceedances":    exceedances,
        "p_expected":     p_expected,
        "p_actual":       round(p_actual, 5),
        "lr_statistic":   round(lr_stat, 4),
        "p_value":        round(p_value, 4),
        "reject_h0":      reject_h0,
        "interpretation": ("⚠️  VaR MAL ESPECIFICADO (rechaza H0 al 5%)"
                           if reject_h0 else
                           "✅  VaR bien especificado (no rechaza H0 al 5%)"),
    }
