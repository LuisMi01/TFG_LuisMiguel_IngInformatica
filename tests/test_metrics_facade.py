"""
Test de la fachada ``RiskMetrics``.

CLAUDE.md sección 4: las funciones puras son la fuente de verdad, y la
fachada las reexporta como ``staticmethod``. No debe duplicar lógica:
cada método de ``RiskMetrics`` debe ser literalmente la misma función pura.
"""
from riskpkg.metrics import (
    RiskMetrics,
    expected_shortfall,
    full_report,
    kupiec_test,
    marginal_risk_contribution,
    max_drawdown,
    sharpe_ratio,
    var_historical,
    var_parametric,
    volatility,
)


def test_facade_reexporta_mismas_funciones_puras():
    """Los staticmethod de RiskMetrics son las MISMAS funciones puras."""
    # __func__ desempaqueta el staticmethod, así comparamos las funciones reales.
    assert RiskMetrics.volatility is volatility
    assert RiskMetrics.var_parametric is var_parametric
    assert RiskMetrics.var_historical is var_historical
    assert RiskMetrics.expected_shortfall is expected_shortfall
    assert RiskMetrics.sharpe_ratio is sharpe_ratio
    assert RiskMetrics.max_drawdown is max_drawdown
    assert RiskMetrics.kupiec_test is kupiec_test
    assert RiskMetrics.marginal_risk_contribution is marginal_risk_contribution
    assert RiskMetrics.full_report is full_report


def test_facade_misma_salida_que_funcion_pura(daily_returns):
    """Llamada via fachada y llamada directa producen el mismo float bit-a-bit."""
    assert RiskMetrics.volatility(daily_returns) == volatility(daily_returns)
    assert RiskMetrics.var_parametric(daily_returns, 0.95) == var_parametric(daily_returns, 0.95)


def test_full_report_devuelve_claves_esperadas(daily_returns):
    rep = full_report(daily_returns, label="TEST")
    claves = {
        "label", "annual_return", "volatility_ann",
        "var_param_95", "var_param_99",
        "var_hist_95", "var_hist_99",
        "expected_shortfall", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "recovery_days",
    }
    assert claves.issubset(rep.keys())
    assert rep["label"] == "TEST"


def test_full_report_via_facade_misma_salida(daily_returns):
    """RiskMetrics.full_report ≡ metrics.full_report."""
    rep_facade = RiskMetrics.full_report(daily_returns, "X")
    rep_direct = full_report(daily_returns, "X")
    assert rep_facade.keys() == rep_direct.keys()
    for k in rep_facade:
        assert rep_facade[k] == rep_direct[k] or (
            # Tolerar NaN igual en ambos lados
            rep_facade[k] != rep_facade[k] and rep_direct[k] != rep_direct[k]
        )
