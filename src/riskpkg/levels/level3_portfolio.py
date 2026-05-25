# -*- coding: utf-8 -*-
"""
NIVEL 3 — Cartera diversificada con benchmark, IA y simulación Monte Carlo.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.constants import (
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    BENCHMARK_TICKER,
    TRADING_DAYS_YEAR,
)
from ..data.loader import DataLoader
from ..metrics import RiskMetrics
from ..models import AI_ModelingLayer


class Level3_PortfolioAnalyzer:
    """
    NIVEL 3: Análisis del riesgo de una cartera diversificada.

    Extiende el Nivel 2 con:
      · Análisis frente a benchmark (SPY por defecto):
        Alpha Jensen, Beta, R², Tracking Error, Information Ratio,
        Up/Down Capture
      · Detección de anomalías (Isolation Forest)
      · Random Forest para importancia de factores de riesgo
      · Coherencia RF vs. MRC clásico (Spearman)
      · Simulación de Monte Carlo multivariante (252 días)
      · Test de Kupiec para el VaR de la cartera
    """

    def __init__(
        self,
        tickers: List[str],
        weights: List[float],
        portfolio_name: str = "Cartera",
        benchmark: str = BENCHMARK_TICKER,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        return_type: str = "log",
        mc_sims: int = 1000,
        mc_days: int = TRADING_DAYS_YEAR,
    ):
        assert abs(sum(weights) - 1.0) < 1e-6, "Los pesos deben sumar 1.0"

        self.tickers = tickers
        self.weights = np.array(weights)
        self.portfolio_name = portfolio_name
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.return_type = return_type
        self.mc_sims = mc_sims
        self.mc_days = mc_days

        # Resultados (se rellenan en run())
        self.asset_returns: Optional[pd.DataFrame] = None
        self.port_returns: Optional[pd.Series] = None
        self.bench_returns: Optional[pd.Series] = None
        self._port_metrics: Optional[Dict] = None
        self._bench_metrics: Optional[Dict] = None
        self._corr_matrix: Optional[pd.DataFrame] = None
        self._mrc: Optional[pd.Series] = None
        self._prc: Optional[pd.Series] = None
        self._div_ratio: Optional[float] = None
        self._anomalies: Optional[pd.Series] = None
        self._rf_results: Optional[Dict] = None
        self._spearman: Optional[Dict] = None
        self._mc_results: Optional[Dict] = None
        self._kupiec: Optional[Dict] = None
        self._alpha_beta: Optional[Tuple] = None
        self._tracking_error: Optional[float] = None
        self._info_ratio: Optional[float] = None
        self._r2: Optional[float] = None
        self._up_down_capture: Optional[Tuple] = None

    def run(self) -> "Level3_PortfolioAnalyzer":
        """Carga datos y ejecuta todos los análisis."""

        # ── Datos ──────────────────────────────────────────────────────────
        loader = DataLoader(
            self.tickers + [self.benchmark],
            self.start_date, self.end_date, self.return_type
        ).fetch()
        all_returns = loader.returns

        # Separar activos de la cartera y benchmark
        tickers_available = [t for t in self.tickers if t in all_returns.columns]
        self.asset_returns = all_returns[tickers_available]

        # Renormalizar pesos
        w = self.weights[:len(tickers_available)]
        w = w / w.sum()
        self.weights_effective = w

        self.bench_returns = all_returns[self.benchmark] if self.benchmark in all_returns.columns else None

        # ── Retornos de cartera ────────────────────────────────────────────
        # Varianza matricial: σ²p = w^T Σ w (más precisa que suma ponderada)
        self.port_returns = (self.asset_returns * w).sum(axis=1)
        self.port_returns.name = self.portfolio_name

        # ── Métricas de riesgo ─────────────────────────────────────────────
        self._port_metrics = RiskMetrics.full_report(self.port_returns, label=self.portfolio_name)
        self._corr_matrix = self.asset_returns.corr()
        self._mrc = RiskMetrics.marginal_risk_contribution(self.asset_returns, w)
        self._prc = RiskMetrics.percentage_risk_contribution(self.asset_returns, w)
        self._div_ratio = RiskMetrics.diversification_ratio(self.asset_returns, w)

        # ── Benchmark ──────────────────────────────────────────────────────
        if self.bench_returns is not None:
            common_idx = self.port_returns.index.intersection(self.bench_returns.index)
            p_aligned = self.port_returns.loc[common_idx]
            b_aligned = self.bench_returns.loc[common_idx]
            self._alpha_beta = RiskMetrics.alpha_beta(p_aligned, b_aligned)
            self._tracking_error = RiskMetrics.tracking_error(p_aligned, b_aligned)
            self._info_ratio = RiskMetrics.information_ratio(p_aligned, b_aligned)
            self._r2 = RiskMetrics.r_squared(p_aligned, b_aligned)
            self._up_down_capture = RiskMetrics.up_down_capture(p_aligned, b_aligned)
            self._bench_metrics = RiskMetrics.full_report(b_aligned, label=self.benchmark)

        # ── Kupiec Test ────────────────────────────────────────────────────
        self._kupiec = RiskMetrics.kupiec_test(self.port_returns, 0.95, "historical")

        # ── Isolation Forest ───────────────────────────────────────────────
        self._anomalies = AI_ModelingLayer.isolation_forest_anomalies(self.port_returns)

        # ── Random Forest + Spearman ───────────────────────────────────────
        self._rf_results = AI_ModelingLayer.rf_risk_factor_importance(
            self.asset_returns, self.port_returns
        )
        self._spearman = AI_ModelingLayer.compare_risk_attribution(
            self._rf_results["rf_importances"],
            self._mrc,
        )

        # ── Monte Carlo ────────────────────────────────────────────────────
        self._mc_results = AI_ModelingLayer.monte_carlo_simulation(
            self.asset_returns, w,
            days=self.mc_days, n_sims=self.mc_sims,
        )

        return self

    def print_report(self) -> None:
        m = self._port_metrics
        ab = self._alpha_beta or (np.nan, np.nan)

        print(f"\n{'═'*60}")
        print(f"  NIVEL 3 — Cartera: {m['label']}")
        print(f"{'═'*60}")
        print(f"  Retorno anual          : {m['annual_return']:>10.2%}")
        print(f"  Volatilidad anual      : {m['volatility_ann']:>10.2%}")
        print(f"  VaR Histórico   (95%)  : {m['var_hist_95']:>10.4f}")
        print(f"  VaR Histórico   (99%)  : {m['var_hist_99']:>10.4f}")
        print(f"  Expected Shortfall     : {m['expected_shortfall']:>10.4f}")
        print(f"  Sharpe Ratio           : {m['sharpe_ratio']:>10.4f}")
        print(f"  Sortino Ratio          : {m['sortino_ratio']:>10.4f}")
        print(f"  Max Drawdown           : {m['max_drawdown']:>10.2%}")
        print(f"  Ratio Diversificación  : {self._div_ratio:>10.4f}")

        # Benchmark
        if self._alpha_beta is not None:
            uc, dc = self._up_down_capture
            print(f"{'─'*60}")
            print(f"  Benchmark: {self.benchmark}")
            print(f"  Alpha Jensen (anual)   : {ab[0]:>10.2%}")
            print(f"  Beta                   : {ab[1]:>10.4f}")
            print(f"  R²                     : {self._r2:>10.4f}")
            print(f"  Tracking Error         : {self._tracking_error:>10.2%}")
            print(f"  Information Ratio      : {self._info_ratio:>10.4f}")
            print(f"  Up Capture Ratio       : {uc:>10.1f}%")
            print(f"  Down Capture Ratio     : {dc:>10.1f}%")

        # Kupiec
        if self._kupiec and "interpretation" in self._kupiec:
            print(f"{'─'*60}")
            k = self._kupiec
            print(f"  Kupiec Test VaR 95%    : {k['interpretation']}")

        # IA: Anomalías
        n_anomalies = (self._anomalies == -1).sum()
        print(f"{'─'*60}")
        print(f"  IA — Isolation Forest")
        print(f"  Días estrés detectados : {n_anomalies} / {len(self.port_returns)} sesiones "
              f"({n_anomalies/len(self.port_returns):.1%})")

        # IA: Random Forest
        print(f"{'─'*60}")
        print(f"  IA — Random Forest (importancia factores de riesgo)")
        for asset, imp in self._rf_results["rf_importances"].items():
            prc_v = self._prc.get(asset, 0)
            print(f"    {asset:>8}  RF: {imp:.3f}  MRC clásico: {prc_v:.1f}%")
        print(f"  Correlación Spearman   : {self._spearman['interpretation']}")
        print(f"  (ρ = {self._spearman['spearman_correlation']:.4f}, "
              f"p = {self._spearman['p_value']:.4f})")

        # Monte Carlo
        mc = self._mc_results
        print(f"{'─'*60}")
        print(f"  Monte Carlo ({mc['n_sims']} sims, {mc['days']} días, seed={mc['seed']})")
        print(f"  Valor inicial          : {mc['initial_value']:>10,.0f} €")
        print(f"  Mediana final (P50)    : {mc['percentiles'][50]:>10,.0f} €")
        print(f"  P5 (peor 5%)           : {mc['percentiles'][5]:>10,.0f} €")
        print(f"  P95 (mejor 5%)         : {mc['percentiles'][95]:>10,.0f} €")
        print(f"  Prob. pérdida          : {mc['prob_loss']:>10.1f}%")
        print(f"{'═'*60}\n")
