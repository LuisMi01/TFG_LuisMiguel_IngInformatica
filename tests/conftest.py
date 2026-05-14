"""
Fixtures sintéticas para la suite de tests de riskpkg.

Toda la generación de datos usa ``np.random.default_rng(RANDOM_SEED)`` para
no contaminar el estado global de NumPy. Ningún test debe tocar red:
yfinance está prohibido en CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskpkg.utils.constants import RANDOM_SEED, TRADING_DAYS_YEAR

# ── Parámetros de mercado sintéticos ────────────────────────────────────────
MU_DAILY = 0.0005      # ≈ 12.6% anualizado
SIGMA_DAILY = 0.012    # ≈ 19% anualizado
N_DAYS_SHORT = 504     # 2 años
N_DAYS_LONG = 1500     # 6 años (Kupiec exige ≥ 250 out-of-sample)


def _business_index(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture
def daily_returns() -> pd.Series:
    """Serie de retornos diarios N(μ, σ) de longitud 504 (2 años)."""
    r = np.random.default_rng(RANDOM_SEED).normal(MU_DAILY, SIGMA_DAILY, N_DAYS_SHORT)
    return pd.Series(r, index=_business_index(N_DAYS_SHORT), name="returns")


@pytest.fixture
def daily_returns_long() -> pd.Series:
    """Serie larga (6 años) para el test de Kupiec."""
    r = np.random.default_rng(RANDOM_SEED + 1).normal(MU_DAILY, SIGMA_DAILY, N_DAYS_LONG)
    return pd.Series(r, index=_business_index(N_DAYS_LONG), name="returns")


@pytest.fixture
def bench_returns() -> pd.Series:
    """Benchmark sintético independiente con misma longitud que ``daily_returns``."""
    r = np.random.default_rng(RANDOM_SEED + 2).normal(MU_DAILY, SIGMA_DAILY, N_DAYS_SHORT)
    return pd.Series(r, index=_business_index(N_DAYS_SHORT), name="benchmark")


@pytest.fixture
def port_and_bench() -> tuple[pd.Series, pd.Series]:
    """
    Par (port, bench) construido con relación lineal conocida:
        port = α_daily + β · bench + ε    con α_anual ≈ 0.03, β = 0.70

    Permite validar la regresión OLS de ``alpha_beta`` contra los coeficientes
    teóricos sin depender de datos de mercado.
    """
    rng_local = np.random.default_rng(RANDOM_SEED + 3)
    bench = rng_local.normal(MU_DAILY, SIGMA_DAILY, N_DAYS_SHORT)
    alpha_daily = 0.03 / TRADING_DAYS_YEAR
    beta = 0.70
    noise = rng_local.normal(0.0, 0.004, N_DAYS_SHORT)
    port = alpha_daily + beta * bench + noise

    idx = _business_index(N_DAYS_SHORT)
    return (
        pd.Series(port, index=idx, name="port"),
        pd.Series(bench, index=idx, name="bench"),
    )


@pytest.fixture
def multiasset_returns() -> pd.DataFrame:
    """
    Retornos de 3 activos parcialmente correlacionados (504 sesiones).
    Matriz de correlación objetivo:
                AAA   BBB   CCC
        AAA   1.00  0.50  0.20
        BBB   0.50  1.00  0.30
        CCC   0.20  0.30  1.00
    """
    rng_local = np.random.default_rng(RANDOM_SEED + 4)
    corr = np.array([
        [1.00, 0.50, 0.20],
        [0.50, 1.00, 0.30],
        [0.20, 0.30, 1.00],
    ])
    sigmas = np.array([SIGMA_DAILY, SIGMA_DAILY * 1.2, SIGMA_DAILY * 0.8])
    cov = corr * np.outer(sigmas, sigmas)
    mean = np.array([MU_DAILY, MU_DAILY * 0.8, MU_DAILY * 1.1])
    samples = rng_local.multivariate_normal(mean, cov, size=N_DAYS_SHORT)
    return pd.DataFrame(
        samples,
        index=_business_index(N_DAYS_SHORT),
        columns=["AAA", "BBB", "CCC"],
    )


@pytest.fixture
def weights3() -> np.ndarray:
    """Pesos de cartera 40/35/25 que suman 1."""
    return np.array([0.40, 0.35, 0.25])


@pytest.fixture
def returns_with_drawdown() -> pd.Series:
    """
    Serie construida con drawdown determinista:
        - 50 sesiones de retornos positivos +0.5%
        - 30 sesiones de retornos negativos -1.0%
        - 70 sesiones de retornos positivos +0.6% (recupera el pico)

    Permite verificar que ``max_drawdown`` es negativo y que
    ``drawdown_recovery_period`` devuelve un entero (recupera) o None.
    """
    parts = [
        np.full(50, 0.005),
        np.full(30, -0.010),
        np.full(70, 0.006),
    ]
    arr = np.concatenate(parts)
    return pd.Series(arr, index=_business_index(len(arr)), name="returns")


@pytest.fixture
def returns_no_recovery() -> pd.Series:
    """Serie con drawdown que no recupera dentro de la propia ventana."""
    parts = [
        np.full(50, 0.005),
        np.full(40, -0.012),
        np.full(10, 0.001),  # rebote insuficiente
    ]
    arr = np.concatenate(parts)
    return pd.Series(arr, index=_business_index(len(arr)), name="returns")
