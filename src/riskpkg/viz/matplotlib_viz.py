# -*- coding: utf-8 -*-
"""
Capa de presentación — Visualizaciones matplotlib/seaborn.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from ..utils.constants import (
    TRADING_DAYS_YEAR,
    DEFAULT_CONFIDENCE,
    DEFAULT_RISK_FREE,
)
from ..metrics.basic import var_historical, expected_shortfall
from ..levels.level4_patrimony import Level4_PatrimonyAnalyzer


class RiskVisualizer:
    """
    Responsabilidad: generar visualizaciones del análisis de riesgo.

    No contiene lógica de negocio. Recibe datos ya calculados.
    Toda visualización usa matplotlib/seaborn y es reproducible.
    """

    PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#795548"]

    @staticmethod
    def plot_cumulative_returns(
        returns: pd.DataFrame,
        title: str = "Retornos Acumulados",
        benchmark_returns: Optional[pd.Series] = None,
    ) -> None:
        """Retornos acumulados de activos y opcionalmente del benchmark."""
        cum = (1 + returns).cumprod()
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, col in enumerate(cum.columns):
            ax.plot(cum.index, cum[col],
                    label=col, linewidth=1.8,
                    color=RiskVisualizer.PALETTE[i % len(RiskVisualizer.PALETTE)])
        if benchmark_returns is not None:
            cum_bench = (1 + benchmark_returns).cumprod()
            ax.plot(cum_bench.index, cum_bench.values,
                    label=benchmark_returns.name or "Benchmark",
                    linewidth=2, linestyle="--", color="black", alpha=0.6)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Valor (base 1)")
        ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_var_distribution(
        returns: pd.Series,
        confidence: float = DEFAULT_CONFIDENCE,
        label: str = "Activo",
    ) -> None:
        """Distribución de retornos con umbral VaR y ES marcados."""
        var_h = var_historical(returns, confidence)
        es = expected_shortfall(returns, confidence)
        kurt = returns.kurt()
        skew = returns.skew()

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(returns, bins=90, alpha=0.65, color="#2196F3",
                edgecolor="none", density=True, label="Distribución empírica")

        # Curva normal de referencia
        x = np.linspace(returns.min(), returns.max(), 300)
        ax.plot(x, norm.pdf(x, returns.mean(), returns.std()),
                "r--", linewidth=1.5, alpha=0.7, label="Distribución normal teórica")

        ax.axvline(-var_h, color="darkorange", linestyle="--", linewidth=2,
                   label=f"VaR Hist. ({confidence:.0%}) = {var_h:.4f}")
        ax.axvline(-es, color="darkred", linestyle="-", linewidth=2,
                   label=f"ES ({confidence:.0%}) = {es:.4f}")

        textstr = f"Curtosis: {kurt:.2f}\nAsimetría: {skew:.2f}"
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

        ax.set_title(f"Distribución de Retornos — {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Retorno diario"); ax.set_ylabel("Densidad")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_correlation_matrix(
        corr_matrix: pd.DataFrame,
        title: str = "Matriz de Correlación",
    ) -> None:
        """Heatmap de correlaciones entre activos."""
        fig, ax = plt.subplots(figsize=(max(6, len(corr_matrix) + 2),
                                        max(5, len(corr_matrix) + 1)))
        mask = np.zeros_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, True)
        sns.heatmap(
            corr_matrix, ax=ax, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1, linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_rolling_metrics(
        port_returns: pd.Series,
        bench_returns: pd.Series,
        window: int = 63,
        title: str = "Métricas Rolling (63 días)",
    ) -> None:
        """Rolling Sharpe, Alpha y Beta deslizantes."""
        rf_d = DEFAULT_RISK_FREE / TRADING_DAYS_YEAR

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # Rolling Sharpe
        roll_sharpe = (
            port_returns.rolling(window).mean() * TRADING_DAYS_YEAR - DEFAULT_RISK_FREE
        ) / (port_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_YEAR))
        axes[0].plot(roll_sharpe, color="#2196F3", linewidth=1.3)
        axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Sharpe Ratio"); axes[0].grid(True, alpha=0.3)

        # Rolling Beta (OLS ventana)
        roll_beta, roll_alpha = [], []
        dates = []
        for i in range(window, len(port_returns)):
            p = port_returns.iloc[i-window:i] - rf_d
            b = bench_returns.iloc[i-window:i] - rf_d
            if len(p) < 30:
                continue
            cov_pb = p.cov(b)
            var_b = b.var()
            beta_t = cov_pb / var_b if var_b != 0 else np.nan
            alpha_t = (p.mean() - beta_t * b.mean()) * TRADING_DAYS_YEAR if not np.isnan(beta_t) else np.nan
            roll_beta.append(beta_t)
            roll_alpha.append(alpha_t)
            dates.append(port_returns.index[i])

        axes[1].plot(dates, roll_beta, color="#FF5722", linewidth=1.3)
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="β=1")
        axes[1].set_ylabel("Beta"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

        axes[2].plot(dates, [a*100 for a in roll_alpha], color="#4CAF50", linewidth=1.3)
        axes[2].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[2].set_ylabel("Alpha (%)"); axes[2].set_xlabel("Fecha"); axes[2].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_monte_carlo(mc_results: Dict, title: str = "Proyección Monte Carlo") -> None:
        """Visualización de las trayectorias Monte Carlo con bandas de percentiles."""
        paths = mc_results["paths"]
        daily_pct = mc_results["daily_percentiles"]   # arrays shape (days,) → bandas
        final_pct = mc_results["percentiles"]         # escalares → cuadro stats
        init_val = mc_results["initial_value"]
        days = np.arange(mc_results["days"])

        fig, ax = plt.subplots(figsize=(13, 6))

        # Trayectorias individuales (submuestra)
        n_plot = min(200, paths.shape[0])
        for i in range(n_plot):
            ax.plot(days, paths[i, :], color="lightblue", alpha=0.15, linewidth=0.4)

        # Bandas de percentiles diarios (shape correcta para fill_between)
        ax.fill_between(days, daily_pct[5], daily_pct[95],
                        color="steelblue", alpha=0.25, label="5º–95º percentil")
        ax.fill_between(days, daily_pct[25], daily_pct[75],
                        color="steelblue", alpha=0.40, label="25º–75º percentil")
        ax.plot(days, daily_pct[50], color="darkblue", linewidth=2.0, label="Mediana (P50)")

        ax.axhline(init_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"Valor inicial: {init_val:,.0f} €")

        # Cuadro de estadísticas (usa percentiles sobre valores finales)
        stats_txt = (
            f"Mediana final: {final_pct[50]:,.0f} €\n"
            f"P5 (peor 5%):  {final_pct[5]:,.0f} €\n"
            f"P95:           {final_pct[95]:,.0f} €\n"
            f"Prob. pérdida: {mc_results['prob_loss']:.1f}%\n"
            f"Sims: {mc_results['n_sims']} | Seed: {mc_results['seed']}"
        )
        ax.text(0.98, 0.04, stats_txt, transform=ax.transAxes, fontsize=8.5,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

        ax.set_title(f"{title} ({mc_results['n_sims']:,} escenarios, "
                     f"{mc_results['days']} días)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Días de negociación"); ax.set_ylabel("Valor de cartera (€)")
        ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_risk_attribution(
        mrc: pd.Series,
        prc: pd.Series,
        rf_importances: Optional[pd.Series] = None,
        title: str = "Atribución del Riesgo",
    ) -> None:
        """Contribución al riesgo clásica vs. importancia Random Forest."""
        ncols = 3 if rf_importances is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        colors = RiskVisualizer.PALETTE[:len(mrc)]

        # MRC (absoluta)
        axes[0].bar(mrc.index, mrc.values, color=colors, alpha=0.85, edgecolor="white")
        axes[0].set_title("Contribución Marginal al Riesgo (MRC)")
        axes[0].set_ylabel("Volatilidad anualizada aportada")
        axes[0].tick_params(axis="x", rotation=25); axes[0].grid(axis="y", alpha=0.3)

        # PRC (porcentual)
        axes[1].bar(prc.index, prc.values, color=colors, alpha=0.85, edgecolor="white")
        axes[1].set_title("Contribución % al Riesgo (PRC)")
        axes[1].set_ylabel("% del riesgo total de cartera")
        axes[1].tick_params(axis="x", rotation=25); axes[1].grid(axis="y", alpha=0.3)

        # RF Importances
        if rf_importances is not None:
            axes[2].bar(rf_importances.index, rf_importances.values,
                        color=colors, alpha=0.85, edgecolor="white")
            axes[2].set_title("RF Feature Importance\n(proxy no lineal)")
            axes[2].set_ylabel("Importancia relativa")
            axes[2].tick_params(axis="x", rotation=25); axes[2].grid(axis="y", alpha=0.3)

        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_anomalies(
        returns: pd.Series,
        anomalies: pd.Series,
        title: str = "Detección de Anomalías (Isolation Forest)",
    ) -> None:
        """Serie de retornos con las anomalías detectadas resaltadas."""
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(returns.index, returns.values, color="#2196F3",
                linewidth=0.9, alpha=0.8, label="Retornos diarios")
        mask_anom = anomalies == -1
        ax.scatter(returns.index[mask_anom], returns.values[mask_anom],
                   color="red", s=18, zorder=5, label=f"Anomalías ({mask_anom.sum()})")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Retorno diario")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_patrimony_breakdown(results: Dict, title: str = "Desglose Patrimonial") -> None:
        """Gráficos de tarta: descomposición por clase de activo y por liquidez."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        def autopct_fmt(pct):
            return f"{pct:.1f}%" if pct > 3 else ""

        # Por clase
        by_class = results["by_class"]
        axes[0].pie(
            list(by_class.values()),
            labels=[Level4_PatrimonyAnalyzer.ASSET_CLASS_LABELS.get(k, k) for k in by_class.keys()],
            autopct=autopct_fmt, startangle=90,
            colors=RiskVisualizer.PALETTE[:len(by_class)],
        )
        axes[0].set_title("Por Clase de Activo")

        # Por liquidez
        by_liq = results["by_liquidity"]
        axes[1].pie(
            list(by_liq.values()),
            labels=[Level4_PatrimonyAnalyzer.LIQUIDITY_LABELS.get(k, k) for k in by_liq.keys()],
            autopct=autopct_fmt, startangle=90,
            colors=["#4CAF50", "#FF9800", "#F44336"][:len(by_liq)],
        )
        axes[1].set_title("Por Nivel de Liquidez")

        plt.tight_layout(); plt.show()

    # ── Stress testing visualizations ─────────────────────────────────────
    @staticmethod
    def plot_historical_stress_path(result: Dict, title: Optional[str] = None) -> None:
        """Senda diaria del valor de la cartera durante una ventana histórica."""
        path = result["portfolio_path"]
        initial = result["initial_value"]
        min_val = result["portfolio_value_min"]
        final_val = result["portfolio_value_final"]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(path.index, path.values, color="#1f4e79", linewidth=1.8,
                label="Valor de cartera")
        ax.axhline(initial, color="gray", linestyle="--", linewidth=1.0,
                   label=f"Valor inicial: {initial:,.0f} €")
        ax.fill_between(path.index, path.values, initial,
                        where=(path.values < initial),
                        color="#FF5722", alpha=0.18, label="Pérdida")

        # Anotar mínimo
        trough_date = path.idxmin()
        ax.scatter([trough_date], [min_val], color="darkred", zorder=5, s=60)
        ax.annotate(f"Mínimo: {min_val:,.0f} €",
                    xy=(trough_date, min_val), xytext=(15, -25),
                    textcoords="offset points", fontsize=9,
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))

        scn_name = result.get("scenario_name", "escenario")
        ax.set_title(title or f"Stress Test Histórico — {scn_name}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Valor de cartera (€)")
        ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)

        # Cuadro resumen
        stats = (
            f"P&L total: {result['portfolio_pnl_pct']:+.2%}\n"
            f"Max DD:    {result['max_drawdown_pct']:+.2%}\n"
            f"Días suelo: {result['days_to_trough']}\n"
            f"Final:     {final_val:,.0f} €"
        )
        ax.text(0.02, 0.04, stats, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_historical_stress_battery(battery: pd.DataFrame,
                                       title: str = "Batería de Stress Tests Históricos") -> None:
        """Barra comparativa de P&L y Max DD para múltiples escenarios."""
        # Filtra filas sin error
        b = battery[battery["pnl_pct"].notna()] if "pnl_pct" in battery.columns else battery

        fig, ax = plt.subplots(figsize=(12, max(4, len(b) * 0.6)))
        y = np.arange(len(b))
        ax.barh(y - 0.2, b["pnl_pct"].values * 100, height=0.4,
                color="#FF5722", alpha=0.85, edgecolor="white", label="P&L total")
        ax.barh(y + 0.2, b["max_drawdown_pct"].values * 100, height=0.4,
                color="#1f4e79", alpha=0.85, edgecolor="white", label="Max Drawdown")
        ax.set_yticks(y)
        ax.set_yticklabels(b.index, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("% de valor de cartera")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_reverse_stress_curve(curve: pd.DataFrame,
                                  title: str = "Curva de Plausibilidad — Reverse Stress") -> None:
        """Pérdida objetivo vs. distancia de Mahalanobis (plausibilidad)."""
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(curve.index * 100, curve["mahalanobis_distance"],
                marker="o", color="#1f4e79", linewidth=2)
        # Bandas de interpretación
        ax.axhspan(0, 2, color="#4CAF50", alpha=0.08, label="Plausible (< 2σ)")
        ax.axhspan(2, 3, color="#FFC107", alpha=0.10, label="Cola moderada (2–3σ)")
        ax.axhspan(3, 4, color="#FF9800", alpha=0.10, label="Extremo (3–4σ)")
        ax.axhspan(4, max(5, curve["mahalanobis_distance"].max() + 1),
                   color="#F44336", alpha=0.10, label="Tail extremo (> 4σ)")
        ax.set_xlabel("Pérdida objetivo (% del valor de cartera)")
        ax.set_ylabel("Distancia de Mahalanobis (σ de cartera)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()
