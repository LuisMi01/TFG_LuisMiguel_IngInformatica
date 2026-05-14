"""Tests de métricas comparativas frente a benchmark."""
import math

import pytest

from riskpkg.metrics.benchmark import (
    HAS_STATSMODELS,
    alpha_beta,
    information_ratio,
    r_squared,
    tracking_error,
    up_down_capture,
)


def test_alpha_beta_recupera_coeficientes_teoricos(port_and_bench):
    """
    Construimos port = α + β·bench + ε con β=0.70 y α≈3% anual.
    OLS sobre 504 puntos debe recuperarlos con tolerancia razonable.
    """
    port, bench = port_and_bench
    alpha, beta = alpha_beta(port, bench, risk_free=0.0)
    assert beta == pytest.approx(0.70, abs=0.05)
    assert alpha == pytest.approx(0.03, abs=0.05)


def test_tracking_error_cero_si_replica_perfecta(daily_returns):
    """Si port == bench, el tracking error es 0."""
    assert tracking_error(daily_returns, daily_returns) == 0.0


def test_tracking_error_es_no_negativo(port_and_bench):
    port, bench = port_and_bench
    assert tracking_error(port, bench) >= 0


def test_information_ratio_nan_si_replica_perfecta(daily_returns):
    """Tracking error == 0 → IR indefinido (NaN)."""
    ir = information_ratio(daily_returns, daily_returns)
    assert math.isnan(ir)


def test_r_squared_uno_si_replica_perfecta(daily_returns):
    assert r_squared(daily_returns, daily_returns) == pytest.approx(1.0, abs=1e-12)


def test_r_squared_en_rango_cero_uno(port_and_bench):
    port, bench = port_and_bench
    r2 = r_squared(port, bench)
    assert 0.0 <= r2 <= 1.0


def test_up_down_capture_devuelve_dos_floats(port_and_bench):
    port, bench = port_and_bench
    up, down = up_down_capture(port, bench)
    assert isinstance(up, float)
    assert isinstance(down, float)


def test_has_statsmodels_es_bool():
    """Solo nos importa que el flag exista y sea booleano (no su valor)."""
    assert isinstance(HAS_STATSMODELS, bool)
