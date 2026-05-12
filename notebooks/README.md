# Notebooks — Demos y validaciones del sistema

Este directorio contiene los notebooks de demostración y validación del paquete
`riskpkg`. Cada notebook tiene una temática clara y reproduce de forma
documentada un caso de uso del sistema.

## Cómo abrirlos en Google Colab

Los ficheros `.py` con docstrings se interpretan automáticamente como notebooks
en Colab cuando se abren mediante:

1. **Vía GitHub directa**: `https://colab.research.google.com/github/<usuario>/<repo>/blob/main/notebooks/01_demo_completo.py`
2. **Carga manual**: en Colab → `Archivo > Cargar notebook` → seleccionar el `.py`.
3. **Conversión a `.ipynb`** (opcional, en local con jupytext):
   ```bash
   pip install jupytext
   jupytext --to notebook 01_demo_completo.py
   ```

Una vez abierto, descomenta la celda de **Colab bootstrap** al inicio del
notebook para clonar el repositorio e instalar el paquete en modo editable.

## Notebooks disponibles

| Notebook | Contenido |
|----------|-----------|
| `01_demo_completo.py` | Demo integral de los 4 niveles (réplica del monolito v2) |
| `06_stress_testing.py` | Stress testing (histórico, hipotético, reverse) con catálogos EBA y CCAR |

## Notebooks planificados (próximas iteraciones)

- `02_nivel1_activo.py` — Análisis detallado de un activo individual.
- `03_nivel2_fondo.py` — Diversificación y MRC por componente.
- `04_nivel3_cartera.py` — Benchmark, IA y Monte Carlo en profundidad.
- `05_nivel4_patrimonio.py` — Patrimonio global con casos representativos.
- `07_validacion_kupiec.py` — Backtesting riguroso del VaR.
