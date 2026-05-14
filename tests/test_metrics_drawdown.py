"""Tests de drawdown y período de recuperación."""
import numpy as np
import pandas as pd

from riskpkg.metrics.drawdown import drawdown_recovery_period, max_drawdown


def test_max_drawdown_serie_monotona_creciente_es_cero():
    """Sin caídas, el drawdown máximo es cero (o casi)."""
    r = pd.Series(np.full(100, 0.001))   # +0.1% constante
    assert max_drawdown(r) == 0.0


def test_max_drawdown_es_negativo_si_hay_caida(returns_with_drawdown):
    assert max_drawdown(returns_with_drawdown) < 0


def test_max_drawdown_oraculo_cerrado():
    """
    Caída controlada: pico 1.10 → mínimo 1.10·(0.9) = 0.99.
    DD = 0.99/1.10 - 1 = -0.10 aprox.
    """
    parts = [np.full(10, 0.00958),    # cumprod ≈ 1.10 (10·0.00958 ≈ 9.58%)
             np.full(10, -0.01047)]   # cumprod desde 1.10 ≈ 0.99
    r = pd.Series(np.concatenate(parts))
    dd = max_drawdown(r)
    # Cota laxa: el drawdown está entre -8% y -12%
    assert -0.12 < dd < -0.08


def test_drawdown_recovery_devuelve_entero_si_recupera(returns_with_drawdown):
    rec = drawdown_recovery_period(returns_with_drawdown)
    assert rec is not None
    assert isinstance(rec, int)
    assert rec >= 0


def test_drawdown_recovery_devuelve_none_si_no_recupera(returns_no_recovery):
    assert drawdown_recovery_period(returns_no_recovery) is None
