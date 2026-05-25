# -*- coding: utf-8 -*-
"""
Capa de lógica de negocio — Métricas de riesgo.

Este paquete expone tanto las funciones puras (recomendado para código nuevo)
como una clase fachada ``RiskMetrics`` que mantiene compatibilidad con la
API original del monolito v2. Los niveles funcionales (Level1-4) usan la
fachada; el código nuevo debería importar las funciones directamente.
"""
from .basic import (
    volatility,
    var_parametric,
    var_historical,
    expected_shortfall,
    sharpe_ratio,
    sortino_ratio,
)
from .drawdown import max_drawdown, drawdown_recovery_period
from .backtest import kupiec_test
from .benchmark import (
    alpha_beta,
    tracking_error,
    information_ratio,
    up_down_capture,
    r_squared,
    HAS_STATSMODELS,
)
from .attribution import (
    diversification_ratio,
    diversification_benefit,
    marginal_risk_contribution,
    percentage_risk_contribution,
)
from .reporter import full_report


class RiskMetrics:
    """
    Fachada de compatibilidad para la clase RiskMetrics del monolito v2.

    Reexpone todas las funciones puras del paquete ``metrics`` como métodos
    estáticos / de clase. Permite que el código existente (Level1-4) siga
    funcionando sin cambios mientras se completa la migración a funciones
    directas en código nuevo.

    Equivalencias:
        RiskMetrics.volatility(r)          ≡  metrics.volatility(r)
        RiskMetrics.full_report(r, "AAPL") ≡  metrics.full_report(r, "AAPL")
    """

    # Básicas
    volatility            = staticmethod(volatility)
    var_parametric        = staticmethod(var_parametric)
    var_historical        = staticmethod(var_historical)
    expected_shortfall    = staticmethod(expected_shortfall)
    sharpe_ratio          = staticmethod(sharpe_ratio)
    sortino_ratio         = staticmethod(sortino_ratio)

    # Drawdown
    max_drawdown                = staticmethod(max_drawdown)
    drawdown_recovery_period    = staticmethod(drawdown_recovery_period)

    # Backtest
    kupiec_test           = staticmethod(kupiec_test)

    # Benchmark
    alpha_beta            = staticmethod(alpha_beta)
    tracking_error        = staticmethod(tracking_error)
    information_ratio     = staticmethod(information_ratio)
    up_down_capture       = staticmethod(up_down_capture)
    r_squared             = staticmethod(r_squared)

    # Atribución
    diversification_ratio          = staticmethod(diversification_ratio)
    diversification_benefit        = staticmethod(diversification_benefit)
    marginal_risk_contribution     = staticmethod(marginal_risk_contribution)
    percentage_risk_contribution   = staticmethod(percentage_risk_contribution)

    # Informe completo (classmethod en el monolito original; staticmethod aquí
    # porque no usa cls realmente)
    full_report           = staticmethod(full_report)


__all__ = [
    # Funciones puras
    "volatility",
    "var_parametric",
    "var_historical",
    "expected_shortfall",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_recovery_period",
    "kupiec_test",
    "alpha_beta",
    "tracking_error",
    "information_ratio",
    "up_down_capture",
    "r_squared",
    "diversification_ratio",
    "diversification_benefit",
    "marginal_risk_contribution",
    "percentage_risk_contribution",
    "full_report",
    # Fachada
    "RiskMetrics",
    # Flags
    "HAS_STATSMODELS",
]
