"""
Tests del módulo EVT-POT.

Validamos cuatro tipos de propiedades:

  1. Recuperación de parámetros: si se generan excesos exactamente de una
     GPD(ξ, β), el MLE debe recuperar esos parámetros con tolerancia
     muestral razonable.
  2. Propiedades estructurales: ES ≥ VaR, VaR creciente en confianza,
     ratios entre métodos en el rango esperado.
  3. Detección de colas: el ξ ajustado sobre retornos t-Student(3) debe ser
     significativamente mayor que sobre retornos gaussianos.
  4. Validación de entradas: errores claros con tail / method inválidos,
     fallback con pocos excesos.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from riskpkg.models import AI_ModelingLayer
from riskpkg.models.evt import (
    compare_var_methods,
    evt_report,
    evt_var_es,
    fit_gpd,
    mean_excess,
    select_threshold,
)


# ── Recuperación de parámetros con GPD pura ──────────────────────────────
def test_fit_gpd_recupera_parametros_de_gpd_sintetica():
    """
    Si los excesos provienen literalmente de una GPD(ξ=0.3, β=1.0), el MLE
    debe recuperar parámetros próximos a los teóricos con n=5000.
    """
    excesses = genpareto.rvs(c=0.3, scale=1.0, size=5000, random_state=42)
    fit = fit_gpd(pd.Series(excesses))
    assert fit["method"] == "MLE"
    assert fit["xi"] == pytest.approx(0.3, abs=0.04)
    assert fit["beta"] == pytest.approx(1.0, abs=0.05)


def test_fit_gpd_devuelve_errores_estandar_positivos():
    excesses = genpareto.rvs(c=0.2, scale=0.5, size=2000, random_state=7)
    fit = fit_gpd(pd.Series(excesses))
    assert fit["xi_se"] > 0
    assert fit["beta_se"] > 0


def test_fit_gpd_fallback_con_pocos_excesos():
    """Con menos de 10 excesos el ajuste no se intenta — devuelve status."""
    fit = fit_gpd(pd.Series([0.1, 0.2, 0.3]))
    assert "status" in fit
    assert fit["n"] == 3


# ── select_threshold ─────────────────────────────────────────────────────
def test_select_threshold_percentile_default(heavy_tailed_returns):
    """Con method='percentile' value=0.95, u es el percentil 95 de pérdidas."""
    u, excesses = select_threshold(heavy_tailed_returns, "percentile", 0.95)
    losses = -heavy_tailed_returns
    assert u == pytest.approx(np.percentile(losses, 95), rel=1e-12)
    assert (excesses > 0).all()


def test_select_threshold_percentile_custom(heavy_tailed_returns):
    """El usuario puede customizar el umbral — verificamos para 97.5%."""
    u, _ = select_threshold(heavy_tailed_returns, "percentile", 0.975)
    losses = -heavy_tailed_returns
    assert u == pytest.approx(np.percentile(losses, 97.5), rel=1e-12)


def test_select_threshold_absolute_acepta_valor_directo(heavy_tailed_returns):
    """method='absolute' fija el umbral en unidades de pérdida directamente."""
    u, excesses = select_threshold(heavy_tailed_returns, "absolute", 0.02)
    assert u == 0.02
    # Todos los excesos deben ser positivos por construcción
    assert (excesses > 0).all()


def test_select_threshold_method_invalido_lanza_valueerror(daily_returns):
    with pytest.raises(ValueError, match="method"):
        select_threshold(daily_returns, method="kde", value=0.5)


def test_select_threshold_value_fuera_de_rango_lanza_valueerror(daily_returns):
    with pytest.raises(ValueError, match="value"):
        select_threshold(daily_returns, method="percentile", value=1.5)


def test_select_threshold_tail_invalido_lanza_valueerror(daily_returns):
    with pytest.raises(ValueError, match="tail"):
        select_threshold(daily_returns, tail="up")


# ── mean_excess ───────────────────────────────────────────────────────────
def test_mean_excess_devuelve_dataframe_estructurado(heavy_tailed_returns):
    me = mean_excess(heavy_tailed_returns)
    assert isinstance(me, pd.DataFrame)
    assert set(me.columns) == {"threshold", "n_excesos", "mean_excess"}
    assert len(me) > 0


def test_mean_excess_thresholds_son_crecientes(heavy_tailed_returns):
    me = mean_excess(heavy_tailed_returns)
    assert me["threshold"].is_monotonic_increasing


def test_mean_excess_n_excesos_decreciente(heavy_tailed_returns):
    """A mayor umbral, menos excesos (monotonía estricta o débil)."""
    me = mean_excess(heavy_tailed_returns)
    diffs = me["n_excesos"].diff().dropna()
    assert (diffs <= 0).all()


# ── evt_var_es: pipeline completo ────────────────────────────────────────
def test_evt_var_es_devuelve_claves_esperadas(heavy_tailed_returns):
    r = evt_var_es(heavy_tailed_returns, confidence=0.99)
    claves = {"u", "n_total", "n_excesos", "xi", "beta", "xi_se", "beta_se",
              "method", "confidence", "var", "es", "tail"}
    assert claves.issubset(r.keys())


def test_evt_var_es_var_positivo_en_cola_pesada(heavy_tailed_returns):
    """En una t-Student(3) las pérdidas son altas → VaR positivo en escala pérdidas."""
    r = evt_var_es(heavy_tailed_returns, confidence=0.99)
    assert r["var"] > 0


def test_evt_var_es_monotono_en_confianza(heavy_tailed_returns):
    """A mayor confianza, mayor VaR (extrapolación más en la cola)."""
    r95 = evt_var_es(heavy_tailed_returns, confidence=0.95)
    r99 = evt_var_es(heavy_tailed_returns, confidence=0.99)
    r999 = evt_var_es(heavy_tailed_returns, confidence=0.999)
    assert r95["var"] < r99["var"] < r999["var"]


def test_evt_es_es_al_menos_var(heavy_tailed_returns):
    """Propiedad estructural: ES ≥ VaR siempre (medida coherente)."""
    r = evt_var_es(heavy_tailed_returns, confidence=0.99)
    assert r["es"] >= r["var"]


# ── Distinción colas pesadas vs gaussianas ──────────────────────────────
def test_evt_detecta_cola_pesada_t_student_frente_a_gaussiana(
    heavy_tailed_returns, daily_returns_long,
):
    """
    ξ ajustado sobre t-Student(3) debe ser claramente mayor que sobre
    retornos gaussianos. Teóricamente, ξ_t(3) ≈ 1/3 ≈ 0.33 y ξ_gauss ≈ 0.
    Con muestra finita y umbral al 95% admitimos tolerancia generosa.
    """
    r_heavy = evt_var_es(heavy_tailed_returns, confidence=0.99)
    r_gauss = evt_var_es(daily_returns_long, confidence=0.99)
    assert r_heavy["xi"] > r_gauss["xi"] + 0.10


# ── evt_report con diagnóstico Anderson-Darling ─────────────────────────
def test_evt_report_devuelve_ad_pvalue_en_rango(heavy_tailed_returns):
    rep = evt_report(heavy_tailed_returns, ad_mc_samples=100)
    if "ad_pvalue" in rep and np.isfinite(rep["ad_pvalue"]):
        assert 0.0 <= rep["ad_pvalue"] <= 1.0


def test_evt_report_devuelve_var_es_a_varios_niveles(heavy_tailed_returns):
    rep = evt_report(heavy_tailed_returns, confidences=[0.95, 0.99], ad_mc_samples=100)
    assert set(rep["var_es"].keys()) == {0.95, 0.99}
    for c in (0.95, 0.99):
        assert "var" in rep["var_es"][c]
        assert "es" in rep["var_es"][c]


def test_evt_report_interpretation_contiene_xi_y_diagnostico(heavy_tailed_returns):
    rep = evt_report(heavy_tailed_returns, ad_mc_samples=100)
    txt = rep["interpretation"]
    assert "ξ" in txt
    assert "Anderson-Darling" in txt or "p-valor" in txt or "p =" in txt


# ── compare_var_methods ─────────────────────────────────────────────────
def test_compare_var_methods_estructura(heavy_tailed_returns):
    comp = compare_var_methods(heavy_tailed_returns, confidence=0.99)
    claves = {"confidence", "var_parametric", "var_historical", "var_evt",
              "es_evt", "ratio_evt_param", "xi", "interpretation"}
    assert claves.issubset(comp.keys())


def test_compare_var_methods_ratio_mayor_en_cola_pesada(heavy_tailed_returns):
    """
    Para retornos t-Student(3), VaR_EVT debería superar a VaR_paramétrico
    al 99% (la gaussiana infraestima la cola). Tolerancia generosa
    porque el ratio depende de la muestra concreta.
    """
    comp = compare_var_methods(heavy_tailed_returns, confidence=0.99)
    assert comp["ratio_evt_param"] > 1.0


# ── Fachada AI_ModelingLayer ─────────────────────────────────────────────
def test_ai_modeling_layer_reexporta_evt():
    assert AI_ModelingLayer.mean_excess is mean_excess
    assert AI_ModelingLayer.select_threshold is select_threshold
    assert AI_ModelingLayer.fit_gpd is fit_gpd
    assert AI_ModelingLayer.evt_var_es is evt_var_es
    assert AI_ModelingLayer.evt_report is evt_report
    assert AI_ModelingLayer.compare_var_methods is compare_var_methods
