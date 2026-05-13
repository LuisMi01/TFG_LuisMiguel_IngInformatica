# CLAUDE.md — Contexto del proyecto para Claude Code

> Este fichero se lee automáticamente al iniciar cada sesión de Claude Code.
> Sirve para que el agente entienda el proyecto sin que el usuario tenga que
> explicar desde cero qué es y qué convenciones se siguen.

---

## 1. Qué es este proyecto

**Trabajo de Fin de Grado** del Grado en Ingeniería Informática (curso 2025-2026).

**Título**: Sistema de Análisis y Gestión del Riesgo Financiero en Carteras Multi-Activo mediante Técnicas Cuantitativas e Inteligencia Artificial.

**Autor**: Luis Miguel Urbez Villar. Perfil profesional: consultor financiero en banca, así que las referencias a EBA, Basilea, CCAR, FRTB, SREP, etc., son terminología familiar y deben usarse con propiedad.

**Tribunal académico**: 60% perfil tecnológico, 50% perfil cuantitativo. Objetivo declarado: matrícula de honor.

**Deadline**: depósito final ~2 meses desde mayo 2026.

## 2. Arquitectura

Paquete Python instalable (`pip install -e .`) organizado en seis subpaquetes por capa:

```
src/riskpkg/
├── data/       → Acceso a datos          (DataLoader: yfinance + CSV + log/simple returns)
├── metrics/    → Lógica de negocio       (VaR · ES · Sharpe · Kupiec · MRC · Alpha/Beta · diversificación)
├── models/     → Modelado IA             (GARCH · GJR-GARCH · Isolation Forest · RF · Monte Carlo)
├── stress/     → Stress testing          (Histórico · Hipotético EBA/CCAR · Reverse-Mahalanobis)
├── levels/     → Orquestación funcional  (Niveles 1, 2, 3, 4)
├── viz/        → Presentación            (matplotlib + seaborn)
└── utils/      → Constantes globales     (RANDOM_SEED=42, TRADING_DAYS_YEAR=252, etc.)
```

**Niveles funcionales** (modelo de agregación patrimonial progresivo):
1. Activo individual
2. Fondo de inversión (cesta con pesos fijos)
3. Cartera diversificada (+ benchmark + IA + Monte Carlo)
4. Patrimonio global (incluye activos no financieros: inmuebles, bonos no cotizados, alternativos)

## 3. Estado actual (v0.4.0)

Completado y verificado:
- ✅ Refactor del monolito original `risk_system_v2.py` (1.790 líneas) a paquete modular.
- ✅ Fachadas `RiskMetrics` y `AI_ModelingLayer` que reexportan funciones puras → mantienen API histórica sin tocar Level1-4.
- ✅ Módulo `stress` completo: 8 escenarios históricos (dot-com, GFC, Eurozona 2011, China 2015, Volmageddon, COVID, 2022, SVB), 4 shocks predefinidos (EBA Adverse, CCAR Severely Adverse, estanflación, +200pb), reverse stress test con solución cerrada de Mahalanobis (Breuer & Csiszár 2013).
- ✅ Notebooks `01_demo_completo.py` y `06_stress_testing.py` reproducen el sistema completo en Colab.

Pendiente (orden propuesto):
- ⏳ Suite de tests con pytest + GitHub Actions CI
- ⏳ DCC-GARCH (correlaciones dinámicas) — cierra limitación 6.1 de la memoria
- ⏳ EVT-POT (colas pesadas) — cierra limitación 6.2 de la memoria
- ⏳ HMM (regímenes de mercado) + SHAP sobre Random Forest (interpretabilidad)
- ⏳ Dashboard Streamlit desplegado en Streamlit Community Cloud
- ⏳ Actualización de la memoria TFG (Word) reflejando arquitectura modular y módulo stress

## 4. Convenciones de código

- **Python 3.11**, type hints donde aportan claridad (no obsesivos).
- **Reproducibilidad obligatoria**: cualquier componente estocástico debe respetar `RANDOM_SEED=42` definido en `riskpkg.utils.constants`. Ejecutar dos veces el mismo análisis debe dar resultados idénticos bit a bit.
- **Validación de datos**: mínimo `MIN_OBSERVATIONS=252` sesiones antes de aceptar métricas como fiables.
- **Fallback gracioso**: si una librería opcional (arch, statsmodels) no está disponible, el código debe seguir funcionando con un método más simple y emitir aviso, no romper.
- **Docstrings en español** con referencias académicas explícitas (Sharpe 1994, Artzner 1999, Choueifaty & Coignard 2008, Glosten-Jagannathan-Runkle 1993, Kupiec 1995, etc.).
- **Comentarios y prints en español** para coherencia con la memoria del TFG.
- **Sin emojis en el código**, sí en los `print_report` para destacar resultados (✅ / ⚠️).
- **Funciones puras como fuente de verdad**, las clases-fachada (`RiskMetrics`, `AI_ModelingLayer`) reexportan con `staticmethod`. No duplicar lógica.

## 5. Convenciones de notebooks

- Los notebooks están en `notebooks/` como ficheros `.py` con docstrings entre celdas de código. Este formato funciona nativamente en Colab y en Jupyter (vía jupytext).
- **Estilo Colab**: docstrings triple-quote `"""..."""` se renderizan como celdas Markdown cuando se abre el `.py` en Colab.
- Cada notebook empieza con una sección de **Colab bootstrap** comentada (`!git clone ... && %cd ... && !pip install -e .`).
- El primer notebook (`01_demo_completo.py`) es la **referencia canónica** de que todo el sistema funciona end-to-end. Cualquier cambio que rompa su output debe justificarse.

## 6. Comandos útiles

```bash
# Instalar en modo editable (recomendado durante desarrollo)
pip install -e ".[dev,notebooks]"

# Smoke test rápido (verifica imports)
python -c "import riskpkg; print(riskpkg.__version__)"

# Ejecutar el demo completo (requiere conexión para yfinance)
python notebooks/01_demo_completo.py

# Ejecutar el notebook de stress testing
python notebooks/06_stress_testing.py

# Tests (cuando exista la suite pytest)
pytest

# Linting (configurado en pyproject.toml)
ruff check src/ tests/
```

## 7. Convenciones de Git

- Rama principal: `main`. Trabajamos directamente sobre `main` salvo refactors grandes.
- Mensajes de commit en formato Conventional Commits con scope:
  - `feat(stress): añade reverse stress test Mahalanobis-óptimo`
  - `fix(metrics): corrige período de recuperación cuando no hay drawdown`
  - `refactor(viz): separa visualizaciones por tipo de análisis`
  - `docs: actualiza README con módulo stress`
- Bodies de commit en español, viñetas con guion, máximo ~72 caracteres por línea.

## 8. Antes de cambiar código

Pasos obligados antes de proponer un refactor o modificación sustancial:

1. **Leer la memoria del TFG** (`TFG_Borrador_v3.docx` en `/mnt/project/` o equivalente local) para no romper la coherencia con lo que se le ha entregado al tutor.
2. **Verificar que la limitación que se está cerrando** está documentada en la sección 6 de la memoria. Si no lo está, comentarlo con el usuario antes de tocar código.
3. **Preservar el output** de `notebooks/01_demo_completo.py` y `notebooks/06_stress_testing.py` salvo justificación explícita.

## 9. Lo que NO se debe hacer

- No introducir dependencias nuevas sin justificación clara (cada librería extra es un punto de fallo en defensa).
- No modificar `RANDOM_SEED` sin avisar (rompe la reproducibilidad de todos los outputs documentados).
- No reorganizar la arquitectura de 4 capas sin discutirlo: está alineada con el diagrama de la sección 5.2 de la memoria.
- No usar inglés en mensajes de print, docstrings o comentarios.
- No commitear datos descargados (`*.parquet`, `*.csv.gz`, `data_cache/` ya están en `.gitignore`).

---

*Última actualización del fichero: tras añadir el módulo de stress testing (v0.4.0).*
