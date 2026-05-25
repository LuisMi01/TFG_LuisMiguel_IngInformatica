# -*- coding: utf-8 -*-
"""
Modelización GARCH(1,1) y GJR-GARCH(1,1,1) para volatilidad condicional.
Incluye fallback automático a volatilidad histórica si no convergen.
"""
from typing import Dict

import numpy as np
import pandas as pd

from ..utils.constants import TRADING_DAYS_YEAR, MIN_OBSERVATIONS

# Fallback gracioso si arch no está disponible
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("[AVISO] Librería 'arch' no instalada. GARCH no disponible. "
          "Instalar con: pip install arch")


def garch_volatility(
    returns: pd.Series,
    model_type: str = "GARCH",   # "GARCH" | "GJR-GARCH"
) -> Dict:
    """
    Modeliza la volatilidad condicional con GARCH(1,1) o GJR-GARCH(1,1,1).

    GJR-GARCH captura el efecto apalancamiento: shocks negativos
    producen mayor volatilidad que positivos del mismo módulo.
    (Glosten, Jagannathan & Runkle, 1993)

    Si el modelo no converge, retorna fallback con volatilidad histórica.
    """
    if not HAS_ARCH:
        vol_hist = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
        return {
            "model": "Fallback (arch no instalado)",
            "vol_forecast_ann": vol_hist,
            "converged": False,
        }

    # Requiere mínimo de observaciones para convergencia fiable
    if len(returns) < MIN_OBSERVATIONS:
        vol_hist = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
        return {
            "model": f"Fallback (< {MIN_OBSERVATIONS} obs.)",
            "vol_forecast_ann": vol_hist,
            "converged": False,
        }

    try:
        returns_scaled = returns * 100  # escalar facilita convergencia

        if model_type == "GJR-GARCH":
            am = arch_model(returns_scaled, vol="Garch", p=1, o=1, q=1, rescale=False)
        else:
            am = arch_model(returns_scaled, vol="Garch", p=1, q=1, rescale=False)

        res = am.fit(disp="off", options={"maxiter": 500})

        # Predicción de varianza un paso adelante
        forecast = res.forecast(horizon=1, reindex=False)
        var_forecast_scaled = forecast.variance.values[-1, 0]
        vol_forecast_ann = (np.sqrt(var_forecast_scaled) / 100) * np.sqrt(TRADING_DAYS_YEAR)

        return {
            "model":            model_type,
            "aic":              res.aic,
            "bic":              res.bic,
            "params":           dict(res.params),
            "vol_forecast_ann": vol_forecast_ann,
            "vol_hist_ann":     returns.std() * np.sqrt(TRADING_DAYS_YEAR),
            "converged":        True,
        }

    except Exception as e:
        vol_hist = returns.std() * np.sqrt(TRADING_DAYS_YEAR)
        return {
            "model":            f"Fallback ({model_type} no convergió: {str(e)[:40]})",
            "vol_forecast_ann": vol_hist,
            "converged":        False,
        }
