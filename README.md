# Sistema de Análisis y Gestión del Riesgo Financiero en Carteras Multi-Activo

**Trabajo de Fin de Grado** · Grado en Ingeniería Informática · Curso 2025–2026
Autor: **Luis Miguel Urbez Villar**

Sistema modular para el análisis cuantitativo del riesgo financiero en carteras
multi-activo, estructurado en cuatro niveles funcionales progresivos y desarrollado
íntegramente en Python. Combina métricas cuantitativas estándar del sector con
técnicas de inteligencia artificial como capa complementaria al núcleo analítico.

---

## Arquitectura

El paquete se organiza en cuatro capas con responsabilidades estrictamente separadas:

```
src/riskpkg/
├── data/        ← Capa de acceso a datos       (yfinance · CSV · log/simple returns)
├── metrics/     ← Capa de lógica de negocio    (VaR · ES · Sharpe · Kupiec · MRC · Alpha/Beta)
├── models/      ← Capa de modelado IA          (GARCH · GJR-GARCH · Isolation Forest · RF · Monte Carlo · EVT-POT/GPD)
├── stress/      ← Stress testing               (Histórico · Hipotético · Reverse-Mahalanobis)
├── levels/      ← Orquestación funcional       (Niveles 1, 2, 3, 4)
├── viz/         ← Capa de presentación         (matplotlib · seaborn)
└── utils/       ← Constantes y configuración global
```

### Niveles funcionales

| Nivel | Alcance | Componentes principales |
|-------|---------|-------------------------|
| **1** | Activo individual | Volatilidad, VaR/ES paramétrico e histórico, Sharpe/Sortino, drawdown, GARCH(1,1), GJR-GARCH, test de Kupiec |
| **2** | Fondo de inversión | Varianza matricial, ratio de diversificación, MRC/PRC por componente |
| **3** | Cartera diversificada | Análisis vs. benchmark (alpha/beta/R²/TE/IR/capture), Isolation Forest, Random Forest, Spearman, Monte Carlo multivariante |
| **4** | Patrimonio global | Integración con activos no financieros, descomposición por clase y liquidez |

---

## Cómo ejecutar el código

El sistema se puede usar de tres formas complementarias: el **dashboard
Streamlit** (recorrido visual de los cuatro niveles), los **notebooks**
(reproducibilidad académica del TFG) y la **API Python** del paquete
`riskpkg` (uso programático). Todas comparten el mismo motor de cálculo.

### 1 · Requisitos

- Python **3.11** o superior (probado en 3.11, 3.12 y 3.13).
- `pip` y, opcionalmente, `git` para clonar el repositorio.
- Conexión a internet sólo si se quieren descargar precios reales con
  `yfinance`; los notebooks y el dashboard incluyen modo demo offline.

### 2 · Instalación

```bash
git clone https://github.com/LuisMi01/TFG_LuisMiguel_IngInformatica.git
cd TFG_LuisMiguel_IngInformatica

# Entorno virtual aislado (macOS/Linux)
python -m venv .venv && source .venv/bin/activate

# Paquete en modo editable + extras de desarrollo y notebooks
pip install -e ".[dev,notebooks]"
```

> **Windows (PowerShell):** sustituir `source .venv/bin/activate` por
> `.\.venv\Scripts\Activate.ps1`.

Comprobación rápida de que el paquete está instalado:

```bash
python -c "import riskpkg; print(riskpkg.__version__)"
# → 0.5.0
```

### 3 · Modo A — Dashboard web (Streamlit)

Capa de presentación multipágina que orquesta el paquete `riskpkg` sin
añadir lógica financiera. Cubre los cuatro niveles y el módulo de stress.

```bash
# Streamlit no está en las dependencias base; instalarlo aparte:
pip install streamlit

# Arrancar la aplicación
streamlit run web/app.py
```

Tras unos segundos se abre el dashboard en `http://localhost:8501`. La
cartera se configura una sola vez en la **barra lateral** y se propaga a
todas las páginas:

| Página | Contenido |
|--------|-----------|
| **Nivel 1 — Activo** | Volatilidad, VaR/ES, Sharpe/Sortino, drawdown, GARCH/GJR-GARCH, Kupiec |
| **Nivel 2 — Fondo** | Métricas agregadas, ratio de diversificación, MRC/PRC, matriz de correlación |
| **Nivel 3 — Cartera** | Análisis vs. benchmark (α/β/R²/TE/IR), Isolation Forest, Random Forest + Spearman, Monte Carlo |
| **Nivel 4 — Patrimonio** | Integración con activos no financieros, rescalado del notional, descomposición por clase y liquidez |
| **Stress Testing** | 8 escenarios históricos, 4 hipotéticos (EBA, CCAR, estanflación, +200pb), shock personalizado y reverse stress test |

Por defecto la web arranca en **modo demo offline** con la cartera
canónica del TFG (`ITX.MC · AMZN · TTWO · ANA · GLD`, 2018-2024). Para
descargar precios reales basta con cambiar el selector "Origen de datos"
a "Live (yfinance)" en la barra lateral.

### 4 · Modo B — Notebooks

Los notebooks están en formato `.py` con docstrings entre celdas; se
abren nativamente en **Google Colab** (recomendado para el TFG) o en
Jupyter local vía `jupytext`.

**Colab (sin instalación local):**

```python
!git clone https://github.com/LuisMi01/TFG_LuisMiguel_IngInformatica.git
%cd TFG_LuisMiguel_IngInformatica
!pip install -q -e .
```

**Local (con el entorno ya instalado):**

```bash
python notebooks/01_demo_completo.py     # Recorrido completo Nivel 1 → Nivel 4
python notebooks/06_stress_testing.py    # Stress histórico, hipotético y reverse
```

`01_demo_completo.py` es la **referencia canónica de correctitud**:
cualquier output del dashboard o de la API debe coincidir con él bit a
bit cuando se usa la misma cartera.

### 5 · Modo C — Uso programático (API Python)

```python
from riskpkg import Level3_PortfolioAnalyzer, RiskVisualizer

portfolio = Level3_PortfolioAnalyzer(
    tickers=["AAPL", "MSFT", "GLD"],
    weights=[0.4, 0.4, 0.2],
    portfolio_name="Cartera Demo",
    benchmark="SPY",
    start_date="2020-01-01",
    end_date="2024-12-31",
).run()

portfolio.print_report()
RiskVisualizer.plot_monte_carlo(portfolio._mc_results)
```

Las clases-fachada (`RiskMetrics`, `AI_ModelingLayer`) reexportan las
funciones puras y mantienen la API histórica del monolito original.

### 6 · Tests

```bash
pytest                       # Suite completa (109 tests, offline)
pytest -k "stress"           # Subconjunto por palabra clave
pytest --cov=riskpkg         # Con informe de cobertura
```

La CI ejecuta automáticamente la suite sobre **Ubuntu × {3.11, 3.12,
3.13}** y **Windows × 3.11** en cada `push` a `main`.

---

## Estado actual del desarrollo

- [x] Refactorización a paquete modular (v0.3.0)
- [x] Compatibilidad con la API del monolito v2 vía clases fachada
- [x] **Módulo de stress testing (v0.4.0)** — histórico (8 escenarios), hipotético (EBA, CCAR + custom), reverse (Mahalanobis-óptimo)
- [x] **Suite de tests con pytest y CI (v0.5.0)** — 109 tests offline, GitHub Actions matrix Ubuntu × [3.11/3.12/3.13] + Windows × 3.11
- [x] **EVT-POT / GPD (v0.5.0)** — VaR y ES con colas pesadas, MLE+PWM, Anderson-Darling, comparación con métodos clásicos
- [x] **Dashboard Streamlit** — 5 páginas multipágina cableadas a `riskpkg` (Niveles 1-4 + Stress), modo demo offline y modo live

---

## Reproducibilidad

Todos los componentes estocásticos (Random Forest, Isolation Forest,
Monte Carlo) usan la semilla global `RANDOM_SEED = 42` definida en
`riskpkg.utils.constants`. Las ejecuciones sucesivas con los mismos parámetros
producen resultados idénticos bit a bit.

---

## Futuros añadidos técnicos al sistema
- [ ] DCC-GARCH para correlaciones dinámicas
- [ ] HMM para detección de regímenes
- [ ] SHAP sobre Random Forest

---

## Licencia

MIT. Sistema desarrollado con fines académicos en el marco del Trabajo de Fin
de Grado.
