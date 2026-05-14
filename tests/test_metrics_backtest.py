"""Test del backtesting de VaR mediante Kupiec (1995)."""
import numpy as np
import pandas as pd

from riskpkg.metrics.backtest import kupiec_test


def test_kupiec_insuficientes_observaciones_devuelve_status():
    """Por debajo de 250 obs no se ejecuta el test (devuelve dict con status)."""
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    result = kupiec_test(r)
    assert "status" in result


def test_kupiec_devuelve_diccionario_con_claves_esperadas(daily_returns_long):
    result = kupiec_test(daily_returns_long, confidence=0.95, var_method="historical")
    claves = {
        "n_total", "n_test", "exceedances",
        "p_expected", "p_actual",
        "lr_statistic", "p_value", "reject_h0",
        "interpretation",
    }
    assert claves.issubset(result.keys())


def test_kupiec_p_value_en_rango(daily_returns_long):
    result = kupiec_test(daily_returns_long, confidence=0.95)
    assert 0.0 <= result["p_value"] <= 1.0
    assert isinstance(result["reject_h0"], (bool, np.bool_))


def test_kupiec_no_rechaza_h0_con_retornos_normales(daily_returns_long):
    """
    Si los retornos provienen de una normal y el VaR histórico se calibra
    sobre la misma distribución, Kupiec NO debería rechazar H0 al 5% en
    una muestra suficientemente grande. Test estadístico — no determinista
    en general, pero con seed fija es reproducible.
    """
    result = kupiec_test(daily_returns_long, confidence=0.95, var_method="historical")
    # El p-value debería ser razonablemente alto; basta con > 0.01 para
    # descartar rechazos espurios bajo H0.
    assert result["p_value"] > 0.01


def test_kupiec_acepta_metodo_parametrico(daily_returns_long):
    """El parámetro var_method='parametric' también debe funcionar."""
    result = kupiec_test(daily_returns_long, confidence=0.95, var_method="parametric")
    assert "exceedances" in result
