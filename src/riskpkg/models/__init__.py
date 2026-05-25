# -*- coding: utf-8 -*-
"""
Capa de modelado (IA y simulación).

Igual que en ``metrics``, este paquete expone funciones puras + una clase
fachada ``AI_ModelingLayer`` que mantiene la API original del monolito v2.
"""
from .garch import garch_volatility, HAS_ARCH
from .anomaly import isolation_forest_anomalies
from .factor_importance import (
    rf_risk_factor_importance,
    compare_risk_attribution,
)
from .monte_carlo import monte_carlo_simulation
from .evt import (
    mean_excess,
    select_threshold,
    fit_gpd,
    evt_var_es,
    evt_report,
    compare_var_methods,
)


class AI_ModelingLayer:
    """
    Fachada de compatibilidad para AI_ModelingLayer del monolito v2.

    Componentes:
      · GARCH(1,1) y GJR-GARCH(1,1,1) — volatilidad condicional
      · Isolation Forest              — detección de anomalías
      · Random Forest                 — importancia de factores de riesgo
      · Monte Carlo multivariante     — proyección probabilística de cartera
      · EVT-POT (GPD)                 — VaR/ES extremos con colas pesadas

    Todos los métodos son estáticos y no almacenan estado.
    """

    garch_volatility            = staticmethod(garch_volatility)
    isolation_forest_anomalies  = staticmethod(isolation_forest_anomalies)
    rf_risk_factor_importance   = staticmethod(rf_risk_factor_importance)
    compare_risk_attribution    = staticmethod(compare_risk_attribution)
    monte_carlo_simulation      = staticmethod(monte_carlo_simulation)

    # EVT-POT
    mean_excess                 = staticmethod(mean_excess)
    select_threshold            = staticmethod(select_threshold)
    fit_gpd                     = staticmethod(fit_gpd)
    evt_var_es                  = staticmethod(evt_var_es)
    evt_report                  = staticmethod(evt_report)
    compare_var_methods         = staticmethod(compare_var_methods)


__all__ = [
    # Funciones puras
    "garch_volatility",
    "isolation_forest_anomalies",
    "rf_risk_factor_importance",
    "compare_risk_attribution",
    "monte_carlo_simulation",
    "mean_excess",
    "select_threshold",
    "fit_gpd",
    "evt_var_es",
    "evt_report",
    "compare_var_methods",
    # Fachada
    "AI_ModelingLayer",
    # Flags
    "HAS_ARCH",
]
