"""
Tests de la capa de modelado.

Foco: reproducibilidad bit-a-bit con RANDOM_SEED=42 (requisito CLAUDE.md §4)
y fallback gracioso cuando una librería opcional no está disponible.
"""
import numpy as np
import pytest

from riskpkg.models import AI_ModelingLayer
from riskpkg.models.anomaly import isolation_forest_anomalies
from riskpkg.models.factor_importance import (
    compare_risk_attribution,
    rf_risk_factor_importance,
)
from riskpkg.models.garch import HAS_ARCH, garch_volatility
from riskpkg.models.monte_carlo import monte_carlo_simulation
from riskpkg.utils.constants import RANDOM_SEED


# ── Reproducibilidad ─────────────────────────────────────────────────────
def test_monte_carlo_reproducible_con_misma_seed(multiasset_returns, weights3):
    """Dos llamadas con la misma seed deben producir paths idénticos bit-a-bit."""
    r1 = monte_carlo_simulation(multiasset_returns, weights3, days=63, n_sims=100, seed=42)
    r2 = monte_carlo_simulation(multiasset_returns, weights3, days=63, n_sims=100, seed=42)
    np.testing.assert_array_equal(r1["paths"], r2["paths"])
    np.testing.assert_array_equal(r1["final_values"], r2["final_values"])


def test_monte_carlo_distinto_con_seeds_diferentes(multiasset_returns, weights3):
    """Sanity check: con seeds distintas, los paths NO deben coincidir."""
    r1 = monte_carlo_simulation(multiasset_returns, weights3, days=63, n_sims=100, seed=42)
    r2 = monte_carlo_simulation(multiasset_returns, weights3, days=63, n_sims=100, seed=999)
    assert not np.array_equal(r1["paths"], r2["paths"])


def test_monte_carlo_seed_por_defecto_es_random_seed(multiasset_returns, weights3):
    """La seed por defecto debe ser RANDOM_SEED=42 (reproducibilidad global)."""
    r = monte_carlo_simulation(multiasset_returns, weights3, days=63, n_sims=100)
    assert r["seed"] == RANDOM_SEED


def test_monte_carlo_shape_y_claves(multiasset_returns, weights3):
    r = monte_carlo_simulation(multiasset_returns, weights3, days=63, n_sims=100)
    assert r["paths"].shape == (100, 63)
    assert r["final_values"].shape == (100,)
    for p in (5, 25, 50, 75, 95):
        assert p in r["percentiles"]
        assert p in r["daily_percentiles"]
        assert r["daily_percentiles"][p].shape == (63,)


def test_isolation_forest_reproducible(daily_returns):
    """Con random_state fijo, las predicciones deben ser idénticas."""
    a = isolation_forest_anomalies(daily_returns, contamination=0.05, random_state=42)
    b = isolation_forest_anomalies(daily_returns, contamination=0.05, random_state=42)
    np.testing.assert_array_equal(a.values, b.values)


def test_isolation_forest_devuelve_etiquetas_validas(daily_returns):
    """El output de IsolationForest es -1 (anomalía) o +1 (normal)."""
    pred = isolation_forest_anomalies(daily_returns, contamination=0.05)
    assert set(pred.unique()).issubset({-1, 1})


def test_rf_factor_importance_reproducible(multiasset_returns):
    """RF con random_state fijo → importancias idénticas entre llamadas."""
    port = (multiasset_returns * np.array([0.4, 0.35, 0.25])).sum(axis=1)
    a = rf_risk_factor_importance(multiasset_returns, port, n_splits=3, random_state=42)
    b = rf_risk_factor_importance(multiasset_returns, port, n_splits=3, random_state=42)
    np.testing.assert_array_equal(
        a["rf_importances"].values, b["rf_importances"].values
    )


def test_rf_factor_importance_devuelve_estructura_esperada(multiasset_returns):
    port = (multiasset_returns * np.array([0.4, 0.35, 0.25])).sum(axis=1)
    r = rf_risk_factor_importance(multiasset_returns, port, n_splits=3)
    assert "rf_importances" in r
    assert "rf_ranking" in r
    assert r["n_cv_splits"] == 3
    # Las importancias de un RF normalizan a 1
    assert r["rf_importances"].sum() == pytest.approx(1.0, abs=1e-9)


def test_compare_risk_attribution_devuelve_spearman(multiasset_returns):
    """Compara dos rankings idénticos → Spearman = 1."""
    port = (multiasset_returns * np.array([0.4, 0.35, 0.25])).sum(axis=1)
    rf = rf_risk_factor_importance(multiasset_returns, port, n_splits=3)
    rf_imp = rf["rf_importances"]
    # Comparamos contra sí misma → corr 1.0
    comp = compare_risk_attribution(rf_imp, rf_imp)
    assert comp["spearman_correlation"] == pytest.approx(1.0, abs=1e-9)
    assert bool(comp["meets_criterion"]) is True


# ── GARCH y fallback gracioso ────────────────────────────────────────────
def test_garch_devuelve_diccionario_con_converged(daily_returns_long):
    """Tras arch>=6.3 instalado, GARCH(1,1) debe ejecutarse y devolver dict."""
    result = garch_volatility(daily_returns_long, model_type="GARCH")
    assert "converged" in result
    assert "vol_forecast_ann" in result
    assert result["vol_forecast_ann"] > 0


def test_garch_fallback_por_pocas_observaciones():
    """Con menos de MIN_OBSERVATIONS=252 obs, GARCH cae a fallback."""
    import pandas as pd
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    result = garch_volatility(r, model_type="GARCH")
    assert result["converged"] is False
    assert "vol_forecast_ann" in result
    assert result["vol_forecast_ann"] > 0


def test_garch_acepta_gjr_garch(daily_returns_long):
    """El modelo alternativo GJR-GARCH también debe ser invocable."""
    result = garch_volatility(daily_returns_long, model_type="GJR-GARCH")
    assert "vol_forecast_ann" in result


def test_has_arch_es_bool():
    assert isinstance(HAS_ARCH, bool)


# ── Fachada AI_ModelingLayer ─────────────────────────────────────────────
def test_ai_modeling_layer_reexporta_funciones_puras():
    assert AI_ModelingLayer.garch_volatility is garch_volatility
    assert AI_ModelingLayer.isolation_forest_anomalies is isolation_forest_anomalies
    assert AI_ModelingLayer.rf_risk_factor_importance is rf_risk_factor_importance
    assert AI_ModelingLayer.compare_risk_attribution is compare_risk_attribution
    assert AI_ModelingLayer.monte_carlo_simulation is monte_carlo_simulation
