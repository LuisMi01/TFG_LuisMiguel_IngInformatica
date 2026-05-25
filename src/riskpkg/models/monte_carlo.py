# -*- coding: utf-8 -*-
"""
Simulación de Monte Carlo multivariante para proyección probabilística
del valor futuro de una cartera.
"""
from typing import Dict

import numpy as np
import pandas as pd

from ..utils.constants import TRADING_DAYS_YEAR, RANDOM_SEED


def monte_carlo_simulation(
    returns: pd.DataFrame,
    weights: np.ndarray,
    days: int = TRADING_DAYS_YEAR,
    n_sims: int = 1000,
    initial_value: float = 10_000,
    seed: int = RANDOM_SEED,
) -> Dict:
    """
    Simulación de Monte Carlo multivariante para proyección del valor de cartera.

    Genera n_sims trayectorias de días de trading usando retornos correlacionados
    parametrizados por la media histórica y la matriz de covarianzas.

    Proporciona distribuciones de probabilidad del valor futuro y percentiles.
    Semilla fija para reproducibilidad entre ejecuciones.
    """
    np.random.seed(seed)
    mean_ret = returns.mean().values
    cov_matrix = returns.cov().values
    paths = np.zeros((n_sims, days))

    for i in range(n_sims):
        daily_sims = np.random.multivariate_normal(mean_ret, cov_matrix, days)
        port_daily = daily_sims @ weights
        paths[i, :] = initial_value * np.cumprod(1 + port_daily)

    final_values = paths[:, -1]

    # Percentiles sobre los valores finales (escalares) → para el cuadro de estadísticas
    final_percentiles = {p: np.percentile(final_values, p) for p in [5, 25, 50, 75, 95]}

    # Percentiles por día (arrays de shape=days) → para las bandas del gráfico
    daily_percentiles = {p: np.percentile(paths, p, axis=0) for p in [5, 25, 50, 75, 95]}

    return {
        "paths":              paths,
        "final_values":       final_values,
        "initial_value":      initial_value,
        "percentiles":        final_percentiles,    # escalares (stats box)
        "daily_percentiles":  daily_percentiles,    # arrays (plot bands)
        "mean_final":         np.mean(final_values),
        "std_final":          np.std(final_values),
        "prob_loss":          np.mean(final_values < initial_value) * 100,
        "var_mc_95":          (initial_value - final_percentiles[5]) / initial_value,
        "days":               days,
        "n_sims":             n_sims,
        "seed":               seed,
    }
