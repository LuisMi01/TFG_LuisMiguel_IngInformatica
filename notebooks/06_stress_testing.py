# -*- coding: utf-8 -*-
"""06_stress_testing.ipynb

Stress testing aplicado a una cartera multi-activo: aproximaciones histórica,
hipotética y reverse, con catálogos de escenarios regulatorios (EBA, CCAR).

# Stress Testing — Sistema de Análisis de Riesgo Multi-Activo
---

**Trabajo de Fin de Grado** · Grado en Ingeniería Informática · Curso 2025–2026
Autor: Luis Miguel Urbez Villar

Este notebook implementa las tres aproximaciones clásicas al stress testing
financiero usadas en la práctica bancaria y aceptadas por los reguladores
(EBA, Fed, Banco de España):

| Tipo | Pregunta que responde | Cuándo se usa |
|------|----------------------|---------------|
| **Histórico** | ¿Qué le pasaría a mi cartera si se repitiese la crisis de 2008? | Validación de robustez frente a crisis conocidas |
| **Hipotético** | ¿Y si las bolsas caen un 30% y los tipos suben 200 pb? | Escenarios regulatorios (CCAR, EBA) |
| **Reverse**   | ¿Qué escenario plausible me haría perder un 20%? | Identificación de vulnerabilidades ocultas |

## Marco regulatorio de referencia

- **Comité de Supervisión Bancaria de Basilea**: Principios de Stress Testing (2018).
- **EBA**: 2023 EU-Wide Stress Test Methodological Note.
- **Federal Reserve**: Dodd-Frank Act Stress Test 2024 (DFAST/CCAR).
- **Banco de España**: integra los resultados EBA en el SREP anual.

---

## 1. Bootstrap (descomentar en Colab)
"""

# !git clone https://github.com/LuisMi01/TFG_LuisMiguel_IngInformatica.git
# %cd TFG_LuisMiguel_IngInformatica
# !pip install -q -e .

"""## 2. Imports y configuración"""

import numpy as np
import pandas as pd

from riskpkg import Level3_PortfolioAnalyzer, RiskVisualizer
from riskpkg.stress import (
    # Histórico
    historical_stress_test, historical_stress_battery,
    list_scenarios, HISTORICAL_SCENARIOS,
    # Hipotético
    apply_shock, FactorShock, list_predefined_shocks, PREDEFINED_SHOCKS,
    # Reverse
    reverse_stress_test, reverse_stress_curve,
    # Reporters
    print_historical_report, print_hypothetical_report,
    print_reverse_report, print_historical_battery_report,
)

# Cartera de referencia (idéntica al notebook 01_demo_completo)
TICKERS = ["ITX.MC", "AMZN", "TTWO", "ANA", "GLD"]
WEIGHTS = [0.30, 0.25, 0.15, 0.20, 0.10]
TICKER_TO_CLASS = {
    "ITX.MC": "equity",     # Inditex
    "AMZN":   "equity",     # Amazon
    "TTWO":   "equity",     # Take-Two Interactive
    "ANA":    "equity",     # Acciona
    "GLD":    "commodity",  # SPDR Gold ETF
}
START = "2018-01-01"
END = "2024-12-31"
INITIAL_VALUE = 100_000.0

"""## 3. STRESS TEST HISTÓRICO

Reaplicamos ventanas reales de crisis a la cartera actual. Los activos que no
existían en la fecha del escenario se excluyen y los pesos se renormalizan
sobre los disponibles.

### 3.1 Catálogo de escenarios disponibles
"""

list_scenarios()

"""### 3.2 Caso individual: COVID 2020 (33 sesiones, el crash más rápido de la historia)"""

result_covid = historical_stress_test(
    tickers=TICKERS,
    weights=WEIGHTS,
    scenario="covid_2020",
    initial_value=INITIAL_VALUE,
)
print_historical_report(result_covid)
RiskVisualizer.plot_historical_stress_path(result_covid)

"""### 3.3 Caso individual: 2022 — inflación, Ucrania y subida de tipos

Este escenario es especialmente relevante para una cartera mixta como la nuestra:
la correlación clásica acciones-bonos se rompió y la mayoría de carteras 60/40
sufrieron simultáneamente.
"""

result_2022 = historical_stress_test(
    tickers=TICKERS,
    weights=WEIGHTS,
    scenario="inflation_2022",
    initial_value=INITIAL_VALUE,
)
print_historical_report(result_2022)
RiskVisualizer.plot_historical_stress_path(result_2022)

"""### 3.4 Batería completa: comparativa de todos los escenarios

Algunos escenarios anteriores a 2010 excluirán a TTWO (oferta pública en 1997,
pero con datos limitados en yfinance) y a ANA/ITX si las series son insuficientes.
El reporte indica los activos efectivamente usados en cada ventana.
"""

battery = historical_stress_battery(
    tickers=TICKERS,
    weights=WEIGHTS,
    initial_value=INITIAL_VALUE,
)
print_historical_battery_report(battery)
RiskVisualizer.plot_historical_stress_battery(battery)

"""## 4. STRESS TEST HIPOTÉTICO

Aplicamos shocks instantáneos sobre la cartera. Soportamos dos niveles de
granularidad: por clase de activo (más principial) y override por ticker.

### 4.1 Catálogo de shocks regulatorios predefinidos
"""

list_predefined_shocks()

"""### 4.2 Escenario EBA Adverso 2023

Este es el escenario adverso usado por la EBA en su stress test bancario de 2023.
Aplicarlo a una cartera del lado *buy-side* nos da una idea del impacto que
las autoridades europeas consideran "plausible pero severo".
"""

result_eba = apply_shock(
    weights=WEIGHTS,
    tickers=TICKERS,
    shock="eba_adverse_2023",
    ticker_to_class=TICKER_TO_CLASS,
    initial_value=INITIAL_VALUE,
)
print_hypothetical_report(result_eba)

"""### 4.3 Escenario CCAR Severely Adverse 2024 (Fed)

Más severo que el EBA: replica una recesión global profunda. Es el escenario
"benchmark" que la Fed usa para los grandes bancos americanos.
"""

result_ccar = apply_shock(
    weights=WEIGHTS,
    tickers=TICKERS,
    shock="ccar_severely_adverse_2024",
    ticker_to_class=TICKER_TO_CLASS,
    initial_value=INITIAL_VALUE,
)
print_hypothetical_report(result_ccar)

"""### 4.4 Escenario hipotético a medida

Para definir un shock propio basta con instanciar `FactorShock`. Aquí, un
escenario de tipo "crisis sectorial tech + huida hacia el oro":
"""

custom_shock = FactorShock(
    name="Crisis tech + flight-to-quality",
    class_shocks={"equity": -0.20, "commodity": +0.18},
    ticker_overrides={
        "AMZN": -0.35,    # Amazon penalizada especialmente
        "TTWO": -0.40,    # Gaming también
    },
    description=(
        "Caída concentrada en el sector tecnológico (Amazon -35%, Take-Two -40%) "
        "con huida hacia el oro (+18%). Inditex y Acciona caen el 20% por contagio."
    ),
)
result_custom = apply_shock(
    weights=WEIGHTS,
    tickers=TICKERS,
    shock=custom_shock,
    ticker_to_class=TICKER_TO_CLASS,
    initial_value=INITIAL_VALUE,
)
print_hypothetical_report(result_custom)

"""## 5. REVERSE STRESS TEST

La pregunta cambia de dirección: en lugar de **"qué pérdida produce este escenario"**,
nos preguntamos **"qué escenario produce esta pérdida"**. La respuesta no es única,
así que buscamos el **más plausible**: el de menor distancia de Mahalanobis bajo
la matriz de covarianzas histórica.

### Fundamento matemático

Bajo el modelo gaussiano multivariante con covarianza $\\Sigma$, el shock $s^*$
que minimiza la distancia de Mahalanobis $s^T \\Sigma^{-1} s$ sujeto a una
pérdida objetivo $w^T s = -L$ tiene solución cerrada:

$$
s^* = -L \\cdot \\frac{\\Sigma w}{w^T \\Sigma w}, \\qquad
d_{\\text{Mahal}}(s^*) = \\frac{L}{\\sigma_p}
$$

La distancia es el **número de desviaciones típicas de la cartera** que representa
la pérdida — el z-score bajo normalidad. Referencia: Breuer & Csiszár (2013).

### 5.1 Ejecutamos primero el Nivel 3 para obtener los retornos históricos de la cartera
"""

print("\n>>> Ejecutando análisis de Nivel 3 (necesario para obtener Σ)...")
lvl3 = Level3_PortfolioAnalyzer(
    tickers=TICKERS,
    weights=WEIGHTS,
    portfolio_name="Cartera Demo Internacional",
    benchmark="SPY",
    start_date=START,
    end_date=END,
).run()

"""### 5.2 Reverse stress: ¿qué escenario nos haría perder un 15% anual?"""

result_rev = reverse_stress_test(
    asset_returns=lvl3.asset_returns,
    weights=lvl3.weights_effective,
    target_loss=0.15,
    horizon="annual",
)
print_reverse_report(result_rev)

"""### 5.3 Curva de plausibilidad

Recorremos pérdidas desde el 5% hasta el 50% y trazamos la distancia de
Mahalanobis correspondiente. Esto identifica el **punto de inflexión** donde
la pérdida deja de ser plausible bajo el régimen histórico.
"""

curve = reverse_stress_curve(
    asset_returns=lvl3.asset_returns,
    weights=lvl3.weights_effective,
    losses=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    horizon="annual",
)
print("\nCurva de plausibilidad:")
print(curve.round(4).to_string())
RiskVisualizer.plot_reverse_stress_curve(curve)

"""## 6. Conclusiones

Los tres tipos de stress testing son complementarios:

- **Histórico** valida la cartera contra crisis reales documentadas.
- **Hipotético** permite testear escenarios regulatorios (EBA, CCAR) y
  escenarios a medida.
- **Reverse** identifica vulnerabilidades ocultas: qué combinaciones de
  movimientos plausibles destruirían un porcentaje dado del valor.

Una cartera bien diversificada debería mostrar:
- En **histórico**: pérdidas máximas acotadas y recuperaciones dentro de
  horizonte razonable.
- En **hipotético**: pérdidas EBA Adverso < 25%, CCAR Severo < 40%.
- En **reverse**: distancia de Mahalanobis > 2σ para una pérdida del 20%
  (escenario consistente con régimen normal, no extremo).
"""

print("\n✅  Stress testing completado.")
print("   Tres aproximaciones (histórica + hipotética + reverse) ejecutadas.\n")
