# -*- coding: utf-8 -*-
"""
Stress testing histórico: aplicación de ventanas reales de crisis pasadas
a una cartera con composición y pesos actuales.

Metodología:
    1. Descargar precios de los activos durante la ventana de crisis.
    2. Calcular retornos diarios y la senda acumulada de la cartera con
       los pesos actuales (no los pesos del pasado).
    3. Reportar la pérdida total, el drawdown máximo, el día del suelo y
       el período de recuperación (si lo hubo dentro de la ventana).

Limitaciones (declaradas explícitamente para el TFG):
    - Activos que no existían en la fecha del escenario se excluyen del
      cálculo y los pesos se renormalizan sobre los disponibles.
    - El método asume que los efectos de cartera se acumulan diariamente
      (rebalanceo implícito), lo cual sobreestima ligeramente la pérdida
      en escenarios sin rebalanceo real.
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..data.loader import DataLoader
from .scenarios import HISTORICAL_SCENARIOS, StressWindow


def historical_stress_test(
    tickers: List[str],
    weights: Union[List[float], np.ndarray],
    scenario: Union[str, StressWindow],
    return_type: str = "log",
    initial_value: float = 10_000.0,
) -> Dict:
    """
    Ejecuta un stress test histórico aplicando una ventana de crisis pasada
    a la cartera actual.

    Parameters
    ----------
    tickers : List[str]
        Tickers de los activos de la cartera.
    weights : array-like
        Pesos actuales de cada activo (deben sumar 1).
    scenario : str | StressWindow
        Clave del escenario (ver scenarios.HISTORICAL_SCENARIOS) o instancia
        de StressWindow.
    return_type : str
        "log" o "simple".
    initial_value : float
        Valor de partida de la cartera al inicio de la ventana.

    Returns
    -------
    dict con las siguientes claves:
        - scenario, start, end                     : metadatos
        - tickers_used, tickers_excluded           : tickers efectivamente usados
        - weights_effective                        : pesos renormalizados
        - portfolio_pnl_pct                        : pérdida/ganancia total (%)
        - portfolio_value_final, portfolio_value_min : valores en euros
        - max_drawdown_pct, days_to_trough        : drawdown intra-ventana
        - recovered_within_window                  : True si recuperó el pico
        - asset_pnl_pct                            : dict ticker -> P&L individual
        - portfolio_path                           : Series con el valor diario
    """
    # Resolver escenario
    if isinstance(scenario, str):
        if scenario not in HISTORICAL_SCENARIOS:
            available = ", ".join(HISTORICAL_SCENARIOS.keys())
            raise KeyError(f"Escenario '{scenario}' no encontrado. Disponibles: {available}")
        scn = HISTORICAL_SCENARIOS[scenario]
    else:
        scn = scenario

    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Los pesos deben sumar 1. Suma actual: {weights.sum():.6f}")

    # Descargar datos de la ventana de crisis
    loader = DataLoader(tickers, scn.start, scn.end, return_type=return_type).fetch()
    asset_returns = loader.returns

    # Identificar tickers efectivamente disponibles y renormalizar pesos
    tickers_available = list(asset_returns.columns)
    tickers_excluded = [t for t in tickers if t not in tickers_available]
    weight_map = dict(zip(tickers, weights))
    w_eff = np.array([weight_map[t] for t in tickers_available])
    if w_eff.sum() == 0:
        raise RuntimeError(
            f"Ninguno de los tickers tenía datos durante {scn.start}–{scn.end}."
        )
    w_eff = w_eff / w_eff.sum()

    # Senda diaria de la cartera (compuesta)
    port_returns = (asset_returns * w_eff).sum(axis=1)
    port_path = initial_value * (1 + port_returns).cumprod()

    # Métricas de pérdida
    final_value = float(port_path.iloc[-1])
    pnl_pct = (final_value - initial_value) / initial_value

    # Drawdown intra-ventana
    cumulative = (1 + port_returns).cumprod()
    running_peak = cumulative.expanding().max()
    drawdown_series = (cumulative / running_peak) - 1
    max_dd = float(drawdown_series.min())
    trough_idx = int(drawdown_series.values.argmin())
    days_to_trough = trough_idx

    # ¿Se recupera dentro de la propia ventana?
    peak_before_trough = float(running_peak.iloc[trough_idx])
    post_trough = cumulative.iloc[trough_idx:]
    recovered = bool((post_trough >= peak_before_trough).any())

    # Valor mínimo absoluto
    min_value = float(port_path.min())

    # P&L individual por activo (informativo)
    asset_cum = (1 + asset_returns).cumprod()
    asset_pnl = {t: float(asset_cum[t].iloc[-1] - 1) for t in tickers_available}

    return {
        "scenario":                  scn.key,
        "scenario_name":             scn.name,
        "scenario_description":      scn.description,
        "start":                     scn.start,
        "end":                       scn.end,
        "n_sessions":                len(port_returns),
        "tickers_used":              tickers_available,
        "tickers_excluded":          tickers_excluded,
        "weights_effective":         dict(zip(tickers_available, w_eff)),
        "initial_value":             initial_value,
        "portfolio_value_final":     final_value,
        "portfolio_value_min":       min_value,
        "portfolio_pnl_pct":         pnl_pct,
        "max_drawdown_pct":          max_dd,
        "days_to_trough":            days_to_trough,
        "recovered_within_window":   recovered,
        "asset_pnl_pct":             asset_pnl,
        "portfolio_path":            port_path,
    }


def historical_stress_battery(
    tickers: List[str],
    weights: Union[List[float], np.ndarray],
    scenarios: Optional[List[str]] = None,
    return_type: str = "log",
    initial_value: float = 10_000.0,
) -> pd.DataFrame:
    """
    Ejecuta una batería de stress tests históricos y devuelve un DataFrame
    comparativo con una fila por escenario.

    Si `scenarios` es None, ejecuta todos los del catálogo.
    """
    if scenarios is None:
        scenarios = list(HISTORICAL_SCENARIOS.keys())

    rows = []
    for key in scenarios:
        try:
            r = historical_stress_test(
                tickers, weights, key,
                return_type=return_type, initial_value=initial_value,
            )
            rows.append({
                "scenario":          r["scenario_name"],
                "start":             r["start"],
                "end":               r["end"],
                "n_sessions":        r["n_sessions"],
                "pnl_pct":           r["portfolio_pnl_pct"],
                "max_drawdown_pct":  r["max_drawdown_pct"],
                "days_to_trough":    r["days_to_trough"],
                "recovered":         r["recovered_within_window"],
                "value_final":       r["portfolio_value_final"],
                "n_assets_excluded": len(r["tickers_excluded"]),
            })
        except Exception as e:
            rows.append({
                "scenario":          HISTORICAL_SCENARIOS[key].name,
                "start":             HISTORICAL_SCENARIOS[key].start,
                "end":               HISTORICAL_SCENARIOS[key].end,
                "error":             str(e)[:60],
            })

    return pd.DataFrame(rows).set_index("scenario")
