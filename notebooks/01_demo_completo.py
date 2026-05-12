# -*- coding: utf-8 -*-
"""01_demo_completo.ipynb

Demostración integral del sistema `riskpkg` en sus cuatro niveles funcionales.

Este notebook replica el output del monolito v2 (`risk_system_v2.py`) tras la
refactorización a paquete modular. Está pensado para ejecutarse en Google Colab
o en un entorno Jupyter local con el paquete instalado.

# Sistema de Análisis y Gestión del Riesgo Financiero en Carteras Multi-Activo
---

**Trabajo de Fin de Grado**  ·  Grado en Ingeniería Informática  ·  Curso 2025-2026
Autor: Luis Miguel Urbez Villar

**Arquitectura por capas:**

| Capa | Módulo | Responsabilidad |
|------|--------|-----------------|
| Presentación | `riskpkg.viz` | Visualizaciones matplotlib/seaborn |
| Modelado IA | `riskpkg.models` | GARCH · Isolation Forest · Random Forest · Monte Carlo |
| Lógica de negocio | `riskpkg.metrics` | VaR · ES · Sharpe · Drawdown · Alpha/Beta · Kupiec · MRC |
| Acceso a datos | `riskpkg.data` | yfinance · CSV · log/simple returns |

**Niveles funcionales:**

1. Activo individual
2. Fondo de inversión
3. Cartera diversificada (+benchmark, +IA, +Monte Carlo)
4. Patrimonio global (incluye activos no financieros)

---

## 1. Instalación y configuración

Si se ejecuta en Google Colab:
- Descomenta y ejecuta la celda de clonado del repositorio.
- Instala el paquete en modo editable.

Si se ejecuta localmente con el repo ya clonado, basta con `pip install -e .`
desde la raíz del proyecto.
"""

# ── Colab bootstrap (descomentar en Colab) ────────────────────────────────
# !git clone https://github.com/LuisMi01/TFG_LuisMiguel_IngInformatica.git
# %cd TFG_LuisMiguel_IngInformatica
# !pip install -q -e .

"""## 2. Imports

Todas las clases del sistema están disponibles directamente desde `riskpkg`
gracias al `__init__.py` raíz que reexporta la API pública.
"""

from riskpkg import (
    Level1_AssetAnalyzer,
    Level2_FundAnalyzer,
    Level3_PortfolioAnalyzer,
    Level4_PatrimonyAnalyzer,
    RiskVisualizer,
    TRADING_DAYS_YEAR,
    RANDOM_SEED,
)

print("\n" + "█"*65)
print("  Sistema de Análisis y Gestión del Riesgo Financiero")
print("  Carteras Multi-Activo — Versión 3.0 (modular)")
print("  TFG | Grado en Ingeniería Informática | 2025-2026")
print("█"*65)

"""## 3. Configuración del caso de prueba

Cartera demo mixta internacional: 4 valores cotizados + oro (ETF).
"""

# ── Configuración del caso de prueba ────────────────────────────────────────
TICKERS = ["ITX.MC", "AMZN", "TTWO", "ANA", "GLD"]
WEIGHTS = [0.30, 0.25, 0.15, 0.20, 0.10]
START = "2018-01-01"
END = "2024-12-31"

"""## 4. NIVEL 1 — Análisis de activos individuales

Para cada activo se calculan métricas básicas (retorno, volatilidad, VaR, ES,
Sharpe, Sortino, drawdown), se ajustan modelos GARCH y GJR-GARCH para la
volatilidad condicional, y se ejecuta el test de Kupiec sobre el VaR al 95%.
"""

print("\n>>> [NIVEL 1] Análisis activo individual: Inditex (ITX.MC)")
lvl1 = Level1_AssetAnalyzer("ITX.MC", START, END, return_type="log").run()
lvl1.print_report()
RiskVisualizer.plot_var_distribution(lvl1.returns, label="ITX.MC")
print("\n-------------------------------------------------\n")

print("\n>>> [NIVEL 1] Análisis activo individual: Amazon (AMZN))")
lvl1 = Level1_AssetAnalyzer("AMZN", START, END, return_type="log").run()
lvl1.print_report()
RiskVisualizer.plot_var_distribution(lvl1.returns, label="AMZN")
print("\n-------------------------------------------------\n")

print("\n>>> [NIVEL 1] Análisis activo individual: Take Two (TTWO))")
lvl1 = Level1_AssetAnalyzer("TTWO", START, END, return_type="log").run()
lvl1.print_report()
RiskVisualizer.plot_var_distribution(lvl1.returns, label="TTWO")
print("\n-------------------------------------------------\n")

print("\n>>> [NIVEL 1] Análisis activo individual: Acciona (ANA))")
lvl1 = Level1_AssetAnalyzer("ANA", START, END, return_type="log").run()
lvl1.print_report()
RiskVisualizer.plot_var_distribution(lvl1.returns, label="ANA")
print("\n-------------------------------------------------\n")

print("\n>>> [NIVEL 1] Análisis activo individual: Commodity Gold (GLD))")
lvl1 = Level1_AssetAnalyzer("GLD", START, END, return_type="log").run()
lvl1.print_report()
RiskVisualizer.plot_var_distribution(lvl1.returns, label="GLD")

"""## 5. NIVEL 2 — Análisis de un fondo de inversión

Tratamos los cinco activos previos como un fondo de inversión con pesos fijos.
Se calculan métricas agregadas (varianza matricial), ratio de diversificación,
beneficio de diversificación y la contribución marginal al riesgo (MRC) por activo.
"""

print("\n>>> [NIVEL 2] Análisis fondo de inversión")
lvl2 = Level2_FundAnalyzer(
    tickers=TICKERS,
    weights=WEIGHTS,
    fund_name="Fondo Demo Internacional",
    start_date=START,
    end_date=END,
    return_type="log",
).run()
lvl2.print_report()

"""## 6. NIVEL 3 — Cartera diversificada con benchmark e IA

Extiende el Nivel 2 con:

- Comparación frente a benchmark SPY (alpha, beta, R², tracking error, IR, capture).
- Detección de anomalías con **Isolation Forest** (sesiones de estrés de mercado).
- Importancia de factores de riesgo con **Random Forest** (validación cruzada walk-forward).
- Coherencia entre el RF y el MRC clásico mediante **correlación de Spearman**.
- **Simulación de Monte Carlo** multivariante a 252 días con 1.000 trayectorias.
- **Test de Kupiec** sobre el VaR de la cartera.
"""

print("\n>>> [NIVEL 3] Análisis cartera diversificada + benchmark + IA")
lvl3 = Level3_PortfolioAnalyzer(
    tickers=TICKERS,
    weights=WEIGHTS,
    portfolio_name="Cartera Demo Internacional",
    benchmark="SPY",
    start_date=START,
    end_date=END,
    return_type="log",
    mc_sims=1000,
    mc_days=TRADING_DAYS_YEAR,
).run()
lvl3.print_report()

"""### Visualizaciones del Nivel 3"""

RiskVisualizer.plot_cumulative_returns(
    lvl3.asset_returns,
    title="Retornos Acumulados — Cartera vs Benchmark",
    benchmark_returns=lvl3.bench_returns,
)
RiskVisualizer.plot_correlation_matrix(lvl3._corr_matrix)
RiskVisualizer.plot_risk_attribution(
    lvl3._mrc, lvl3._prc,
    rf_importances=lvl3._rf_results["rf_importances"],
)
RiskVisualizer.plot_anomalies(lvl3.port_returns, lvl3._anomalies)
RiskVisualizer.plot_monte_carlo(lvl3._mc_results)
if lvl3.bench_returns is not None:
    RiskVisualizer.plot_rolling_metrics(lvl3.port_returns, lvl3.bench_returns)

"""## 7. NIVEL 4 — Patrimonio global

Integra la cartera financiera con activos no financieros (inmuebles, bonos no
cotizados, alternativos, efectivo). Estos se modelan con parámetros exógenos
estimados (retorno esperado, volatilidad, correlación con mercado).

Se obtiene una visión patrimonial completa con descomposición por **clase de
activo** y por **nivel de liquidez**.
"""

non_financial_assets = [
    {
        "name":               "Inmueble Madrid",
        "class":              "real_estate",
        "liquidity":          "illiquid",
        "value":              350_000,
        "annual_return_est":  0.03,
        "volatility_est":     0.06,
        "correlation_with_market": 0.25,
    },
    {
        "name":               "Bono Tesoro 10Y",
        "class":              "fixed_income",
        "liquidity":          "semi_liquid",
        "value":              100_000,
        "annual_return_est":  0.035,
        "volatility_est":     0.04,
        "correlation_with_market": -0.20,
    },
    {
        "name":               "Fondo Infraestructuras",
        "class":              "alternative",
        "liquidity":          "illiquid",
        "value":              75_000,
        "annual_return_est":  0.05,
        "volatility_est":     0.08,
        "correlation_with_market": 0.40,
    },
    {
        "name":               "Efectivo / Depósitos",
        "class":              "cash",
        "liquidity":          "liquid",
        "value":              30_000,
        "annual_return_est":  0.025,
        "volatility_est":     0.0,
        "correlation_with_market": 0.0,
    },
]

print("\n>>> [NIVEL 4] Análisis patrimonio global")
lvl4 = Level4_PatrimonyAnalyzer(
    financial_portfolio=lvl3,
    non_financial_assets=non_financial_assets,
    patrimony_name="Patrimonio Global Demo",
).run()
lvl4.print_report()
RiskVisualizer.plot_patrimony_breakdown(lvl4._results)

"""## 8. Cierre

Si todos los outputs anteriores son idénticos a los del monolito v2, la refactorización
modular preserva el comportamiento al 100 %.
"""

print("\n✅  Demo completada correctamente.")
print(f"   Semilla global usada: RANDOM_SEED = {RANDOM_SEED}")
print("   Todos los outputs son reproducibles.\n")
