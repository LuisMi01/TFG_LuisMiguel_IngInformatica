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
├── models/      ← Capa de modelado IA          (GARCH · GJR-GARCH · Isolation Forest · RF · Monte Carlo)
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

## Instalación

### Opción A — Google Colab (recomendado para el TFG)

```python
!git clone https://github.com/LuisMi01/TFG_LuisMiguel_IngInformatica.git
%cd TFG_LuisMiguel_IngInformatica
!pip install -q -e .
```

### Opción B — Local

```bash
git clone https://github.com/LuisMi01/TFG_LuisMiguel_IngInformatica.git
cd TFG_LuisMiguel_IngInformatica
python -m venv .venv && source .venv/bin/activate   # (macOS/Linux)
pip install -e ".[dev,notebooks]"
```

---

## Quickstart

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

Demo integral de los cuatro niveles: ver [`notebooks/01_demo_completo.py`](notebooks/01_demo_completo.py).

---

## Estado actual del desarrollo

- [x] Refactorización a paquete modular (v0.3.0)
- [x] Compatibilidad con la API del monolito v2 vía clases fachada
- [x] **Módulo de stress testing (v0.4.0)** — histórico (8 escenarios), hipotético (EBA, CCAR + custom), reverse (Mahalanobis-óptimo)
- [ ] Suite de tests con pytest y CI
- [ ] DCC-GARCH para correlaciones dinámicas
- [ ] HMM para detección de regímenes
- [ ] SHAP sobre Random Forest
- [ ] Dashboard Streamlit

---

## Reproducibilidad

Todos los componentes estocásticos (Random Forest, Isolation Forest,
Monte Carlo) usan la semilla global `RANDOM_SEED = 42` definida en
`riskpkg.utils.constants`. Las ejecuciones sucesivas con los mismos parámetros
producen resultados idénticos bit a bit.

---

## Licencia

MIT. Sistema desarrollado con fines académicos en el marco del Trabajo de Fin
de Grado.
