# -*- coding: utf-8 -*-
"""
NIVEL 1 — Análisis del riesgo de un activo financiero individual.
"""
from typing import Dict, Optional

import pandas as pd

from ..utils.constants import DEFAULT_START_DATE, DEFAULT_END_DATE
from ..data.loader import DataLoader
from ..metrics import RiskMetrics
from ..models import AI_ModelingLayer


class Level1_AssetAnalyzer:
    """
    NIVEL 1: Análisis del riesgo de un activo financiero individual.

    Métricas implementadas:
      · Retorno anualizado
      · Volatilidad histórica anualizada
      · VaR paramétrico e histórico (95% y 99%)
      · Expected Shortfall (CVaR)
      · Max Drawdown + período de recuperación
      · Sharpe Ratio, Sortino Ratio
      · GARCH(1,1) y GJR-GARCH para comparación de volatilidad condicional
      · Test de Kupiec para backtesting del VaR
    """

    def __init__(
        self,
        ticker: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        return_type: str = "log",
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.return_type = return_type
        self._metrics: Optional[Dict] = None
        self._returns: Optional[pd.Series] = None
        self._garch: Optional[Dict] = None
        self._gjr: Optional[Dict] = None
        self._kupiec: Optional[Dict] = None

    def run(self) -> "Level1_AssetAnalyzer":
        """Carga datos y calcula todas las métricas. Encadenamiento posible."""
        loader = DataLoader(
            [self.ticker], self.start_date, self.end_date, self.return_type
        ).fetch()
        self._returns = loader.returns[self.ticker]

        # Métricas cuantitativas
        self._metrics = RiskMetrics.full_report(self._returns, label=self.ticker)

        # Modelos GARCH
        self._garch = AI_ModelingLayer.garch_volatility(self._returns, "GARCH")
        self._gjr = AI_ModelingLayer.garch_volatility(self._returns, "GJR-GARCH")

        # Backtesting VaR (Kupiec)
        self._kupiec = RiskMetrics.kupiec_test(self._returns, 0.95, "historical")

        return self

    @property
    def metrics(self) -> Dict:
        if self._metrics is None:
            raise RuntimeError("Ejecuta run() primero.")
        return self._metrics

    @property
    def returns(self) -> pd.Series:
        if self._returns is None:
            raise RuntimeError("Ejecuta run() primero.")
        return self._returns

    def print_report(self) -> None:
        m = self.metrics
        print(f"\n{'═'*58}")
        print(f"  NIVEL 1 — Activo Individual: {m['label']}")
        print(f"{'═'*58}")
        print(f"  Retorno anual          : {m['annual_return']:>10.2%}")
        print(f"  Volatilidad anual      : {m['volatility_ann']:>10.2%}")
        print(f"{'─'*58}")
        print(f"  VaR Paramétrico (95%)  : {m['var_param_95']:>10.4f}")
        print(f"  VaR Paramétrico (99%)  : {m['var_param_99']:>10.4f}")
        print(f"  VaR Histórico   (95%)  : {m['var_hist_95']:>10.4f}")
        print(f"  VaR Histórico   (99%)  : {m['var_hist_99']:>10.4f}")
        print(f"  Expected Shortfall     : {m['expected_shortfall']:>10.4f}")
        print(f"{'─'*58}")
        print(f"  Sharpe Ratio           : {m['sharpe_ratio']:>10.4f}")
        print(f"  Sortino Ratio          : {m['sortino_ratio']:>10.4f}")
        print(f"  Max Drawdown           : {m['max_drawdown']:>10.2%}")
        rec = m['recovery_days']
        print(f"  Período recuperación   : "
              f"{'Sin recuperar aún' if rec is None else f'{rec} sesiones':>18}")
        print(f"{'─'*58}")
        if self._garch:
            g = self._garch
            gjr = self._gjr or {}
            print(f"  Vol. Histórica         : {m['volatility_ann']:>10.2%}")
            if g.get("converged"):
                print(f"  Vol. GARCH(1,1)        : {g['vol_forecast_ann']:>10.2%}")
            else:
                print(f"  Vol. GARCH(1,1)        : {g.get('model','N/D')}")
            if gjr.get("converged"):
                print(f"  Vol. GJR-GARCH         : {gjr['vol_forecast_ann']:>10.2%}")
        print(f"{'─'*58}")
        if self._kupiec and "interpretation" in self._kupiec:
            k = self._kupiec
            print(f"  Kupiec Test VaR 95%    : {k['interpretation']}")
            print(f"    Excedencias: {k['exceedances']}/{k['n_test']}  "
                  f"| p-valor: {k['p_value']}")
        print(f"{'═'*58}\n")
