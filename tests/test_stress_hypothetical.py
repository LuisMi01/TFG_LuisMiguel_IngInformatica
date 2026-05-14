"""Tests de stress testing hipotético (shocks por clase de activo)."""
import numpy as np
import pytest

from riskpkg.stress.hypothetical import (
    PREDEFINED_SHOCKS,
    FactorShock,
    apply_shock,
)


# ── Catálogo regulatorio ─────────────────────────────────────────────────
def test_catalogo_contiene_shocks_regulatorios_clave():
    """CLAUDE.md §3: EBA Adverse, CCAR Severely Adverse, estanflación, +200pb."""
    esperados = {
        "eba_adverse_2023",
        "ccar_severely_adverse_2024",
        "stagflation",
        "rate_shock_200bp",
    }
    assert esperados.issubset(set(PREDEFINED_SHOCKS.keys()))


# ── apply_shock: oráculo cerrado ─────────────────────────────────────────
def test_apply_shock_pnl_es_combinacion_lineal_de_shocks_por_clase():
    """
    Con pesos [0.6, 0.4] y shock equity=-0.30, fixed_income=-0.10:
    P&L cartera = 0.6*(-0.30) + 0.4*(-0.10) = -0.22
    """
    pnl = apply_shock(
        weights=[0.6, 0.4],
        tickers=["AAA", "BBB"],
        shock="eba_adverse_2023",
        ticker_to_class={"AAA": "equity", "BBB": "fixed_income"},
        initial_value=10_000.0,
    )
    expected = 0.6 * (-0.30) + 0.4 * (-0.10)
    assert pnl["portfolio_pnl_pct"] == pytest.approx(expected, rel=1e-12)
    assert pnl["portfolio_value_after"] == pytest.approx(
        10_000.0 * (1 + expected), rel=1e-12
    )


def test_apply_shock_default_shock_para_activos_sin_clase():
    """
    Activos sin mapeo a clase reciben default_shock=0.0 (EBA estándar).
    Con pesos [0.5, 0.5] y solo el primero clasificado como equity:
    P&L = 0.5*(-0.30) + 0.5*(0.0) = -0.15
    """
    pnl = apply_shock(
        weights=[0.5, 0.5],
        tickers=["EQ", "UNKNOWN"],
        shock="eba_adverse_2023",
        ticker_to_class={"EQ": "equity"},
    )
    assert pnl["portfolio_pnl_pct"] == pytest.approx(-0.15, rel=1e-12)


def test_apply_shock_ticker_override_tiene_prioridad_sobre_clase():
    """Si un ticker tiene override, no aplica el shock de clase."""
    shock = FactorShock(
        name="custom",
        class_shocks={"equity": -0.30},
        ticker_overrides={"AAA": -0.50},  # más severo que la clase
    )
    pnl = apply_shock(
        weights=[1.0],
        tickers=["AAA"],
        shock=shock,
        ticker_to_class={"AAA": "equity"},
    )
    assert pnl["portfolio_pnl_pct"] == pytest.approx(-0.50, rel=1e-12)


# ── Validaciones de entrada ──────────────────────────────────────────────
def test_apply_shock_pesos_no_suman_uno_lanza_valueerror():
    with pytest.raises(ValueError, match="sumar 1"):
        apply_shock(weights=[0.6, 0.6], tickers=["A", "B"], shock="eba_adverse_2023")


def test_apply_shock_dimension_incompatible_lanza_valueerror():
    with pytest.raises(ValueError, match="misma longitud"):
        apply_shock(weights=[1.0], tickers=["A", "B"], shock="eba_adverse_2023")


def test_apply_shock_clave_inexistente_lanza_keyerror():
    with pytest.raises(KeyError, match="no encontrado"):
        apply_shock(
            weights=[1.0], tickers=["A"],
            shock="this_shock_does_not_exist",
            ticker_to_class={"A": "equity"},
        )


def test_apply_shock_admite_factorshock_directo():
    """Aceptar instancia de FactorShock, no solo string."""
    shock = FactorShock(
        name="manual",
        class_shocks={"equity": -0.20},
    )
    pnl = apply_shock(
        weights=np.array([1.0]),
        tickers=["A"],
        shock=shock,
        ticker_to_class={"A": "equity"},
    )
    assert pnl["portfolio_pnl_pct"] == pytest.approx(-0.20, rel=1e-12)


def test_apply_shock_contribuciones_suman_pnl_total():
    """La suma de contribuciones individuales == pnl agregado en euros."""
    pnl = apply_shock(
        weights=[0.4, 0.35, 0.25],
        tickers=["AAA", "BBB", "CCC"],
        shock="ccar_severely_adverse_2024",
        ticker_to_class={"AAA": "equity", "BBB": "fixed_income", "CCC": "real_estate"},
        initial_value=10_000.0,
    )
    suma_contribs = sum(pnl["asset_contributions"].values())
    assert suma_contribs == pytest.approx(pnl["portfolio_pnl_eur"], rel=1e-10)
