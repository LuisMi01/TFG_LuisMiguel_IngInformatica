# -*- coding: utf-8 -*-
"""
NIVEL 2 — Análisis del riesgo de un fondo de inversión (cesta de activos).
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.constants import DEFAULT_START_DATE, DEFAULT_END_DATE
from ..data.loader import DataLoader
from ..metrics import RiskMetrics


class Level2_FundAnalyzer:
    """
    NIVEL 2: Análisis del riesgo de un fondo de inversión.

    Un fondo = cesta de activos con pesos fijos.
    Reutiliza RiskMetrics sobre retornos ponderados.

    Métricas adicionales respecto al Nivel 1:
      · Varianza de cartera matricial: σ²p = w^T Σ w
      · Ratio de Diversificación
      · Beneficio de Diversificación (puntos porcentuales)
      · Contribución Marginal al Riesgo (MRC) por activo
      · Contribución Porcentual al Riesgo (PRC %)
    """

    def __init__(
        self,
        tickers: List[str],
        weights: List[float],
        fund_name: str = "Fondo",
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        return_type: str = "log",
    ):
        assert len(tickers) == len(weights), "Tickers y pesos deben tener la misma longitud."
        assert abs(sum(weights) - 1.0) < 1e-6, "Los pesos deben sumar 1.0"

        self.tickers = tickers
        self.weights = np.array(weights)
        self.fund_name = fund_name
        self.start_date = start_date
        self.end_date = end_date
        self.return_type = return_type

        self._fund_returns: Optional[pd.Series] = None
        self._asset_returns: Optional[pd.DataFrame] = None
        self._fund_metrics: Optional[Dict] = None
        self._asset_metrics: Optional[Dict[str, Dict]] = None
        self._mrc: Optional[pd.Series] = None
        self._prc: Optional[pd.Series] = None
        self._div_ratio: Optional[float] = None
        self._div_benefit: Optional[float] = None

    def run(self) -> "Level2_FundAnalyzer":
        loader = DataLoader(
            self.tickers, self.start_date, self.end_date, self.return_type
        ).fetch()
        self._asset_returns = loader.returns

        # Renormalizar pesos si algún ticker fue excluido
        available = self._asset_returns.columns.tolist()
        w = self.weights[:len(available)]
        w = w / w.sum()
        self.weights_effective = w

        # Retornos ponderados del fondo
        self._fund_returns = (self._asset_returns * w).sum(axis=1)
        self._fund_returns.name = self.fund_name

        # Métricas del fondo
        self._fund_metrics = RiskMetrics.full_report(
            self._fund_returns, label=self.fund_name
        )

        # Métricas individuales por activo subyacente
        self._asset_metrics = {
            t: RiskMetrics.full_report(self._asset_returns[t], label=t)
            for t in available
        }

        # Diversificación y contribución al riesgo
        self._div_ratio = RiskMetrics.diversification_ratio(self._asset_returns, w)
        self._div_benefit = RiskMetrics.diversification_benefit(self._asset_returns, w)
        self._mrc = RiskMetrics.marginal_risk_contribution(self._asset_returns, w)
        self._prc = RiskMetrics.percentage_risk_contribution(self._asset_returns, w)

        return self

    def print_report(self) -> None:
        m = self._fund_metrics
        print(f"\n{'═'*58}")
        print(f"  NIVEL 2 — Fondo: {m['label']}")
        print(f"{'═'*58}")
        print(f"  Retorno anual          : {m['annual_return']:>10.2%}")
        print(f"  Volatilidad anual      : {m['volatility_ann']:>10.2%}")
        print(f"  VaR Histórico   (95%)  : {m['var_hist_95']:>10.4f}")
        print(f"  Expected Shortfall     : {m['expected_shortfall']:>10.4f}")
        print(f"  Sharpe Ratio           : {m['sharpe_ratio']:>10.4f}")
        print(f"  Max Drawdown           : {m['max_drawdown']:>10.2%}")
        print(f"{'─'*58}")
        print(f"  Ratio de Diversificación   : {self._div_ratio:>8.4f}")
        print(f"  Beneficio Diversificación  : {self._div_benefit:>8.2%}  "
              f"(reducción vol. anualizada)")
        print(f"{'─'*58}")
        print(f"  Contribución al Riesgo por Activo (MRC / PRC):")
        for t in self._asset_returns.columns:
            mrc_v = self._mrc[t]
            prc_v = self._prc[t]
            bar = "█" * max(1, int(prc_v / 5))
            print(f"    {t:>8}  MRC: {mrc_v:.4f}  PRC: {prc_v:>5.1f}%  {bar}")
        print(f"{'─'*58}")
        print(f"  Activos subyacentes:")
        for t, am in self._asset_metrics.items():
            print(f"    [{t:>8}]  Vol: {am['volatility_ann']:.2%}  "
                  f"Sharpe: {am['sharpe_ratio']:.2f}  "
                  f"VaR95h: {am['var_hist_95']:.4f}")
        print(f"{'═'*58}\n")
