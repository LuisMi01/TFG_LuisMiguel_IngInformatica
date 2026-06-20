# Especificación de desarrollo — Capa web (Streamlit) sobre `riskpkg`

> **Cómo usar este documento.** Guárdalo en el repositorio como `docs/WEB_SPEC.md` y, en tu primer mensaje a Claude Code, dile: *"Lee `docs/WEB_SPEC.md`, es la fuente de verdad del alcance. Empieza por la Fase 0 y para al terminarla para que la revise."* También puedes pegar el documento entero como mensaje inicial. Si algo no está aquí, Claude Code debe preguntar antes de implementarlo, nunca improvisar alcance.

---

## 0. Instrucción inicial para Claude Code

Vas a construir la **capa de presentación web** de un sistema de análisis de riesgo financiero cuyo motor de cálculo (`riskpkg`) **ya está terminado, testeado (86 tests, CI en verde) y es la única fuente de verdad**. Tu trabajo NO es escribir lógica financiera: es orquestar llamadas a `riskpkg` y mostrar sus resultados en una interfaz Streamlit.

Tu primer paso, **antes de escribir una sola línea de UI**, es:

1. Explorar el paquete `riskpkg` instalado en el repo (estructura de subpaquetes, clases públicas, funciones, firmas reales y **tipos de retorno**).
2. Generar `docs/API_MAP.md` documentando la API real que vas a consumir: qué clase/función produce cada métrica de cada nivel, qué argumentos recibe y qué devuelve (DataFrame, dict, objeto, figura matplotlib...).
3. Proponerme un plan concreto de la Fase 0 basado en lo que has encontrado, y **esperar mi visto bueno antes de implementar**.

No asumas nombres de funciones ni firmas a partir de este documento: las referencias de API que aparecen abajo son orientativas y debes verificarlas contra el código real.

---

## 1. Objetivo

Construir una aplicación Streamlit que sirva como **front-end de demostración** del sistema para su presentación ante tribunal académico. Debe permitir, de forma interactiva:

- Construir una cartera (tickers, pesos, fechas, benchmark, tasa libre de riesgo) desde la interfaz.
- Recorrer los cuatro niveles funcionales del análisis (activo → fondo → cartera → patrimonio global).
- Ejecutar las tres aproximaciones de stress testing (histórico, hipotético, reverse).
- Visualizar resultados (tablas de métricas + figuras matplotlib) en vivo.

El valor de la web para el TFG es triple: herramienta interactiva para la defensa, artefacto que parece un producto, y entrada por formularios que sustituye la edición manual de diccionarios en notebook (clave en el Nivel 4).

---

## 2. Reglas de oro (NO negociables)

1. **`riskpkg` es la única fuente de verdad.** Cero cálculos financieros en la capa Streamlit. Si necesitas una métrica, **llama a `riskpkg`**. Si no existe esa función en el paquete, **PARA y pregunta** — no la reimplementes en la web. Una segunda implementación divergente del cálculo es el peor error posible en este proyecto.
2. **Explora antes de codificar.** Documenta la API real en `docs/API_MAP.md` antes de construir UI.
3. **Reutiliza las figuras existentes.** Las visualizaciones ya están implementadas en `riskpkg.viz` como figuras matplotlib. Muéstralas con `st.pyplot(fig)`. **No reescribas gráficos** salvo que una vista no exista en `viz`, en cuyo caso pregunta.
4. **Respeta el determinismo.** `RANDOM_SEED = 42` ya está propagado dentro de `riskpkg`. No lo toques, no introduzcas nuevas fuentes de aleatoriedad en la web. Los resultados de la web deben coincidir **bit a bit** con los de los notebooks de demostración.
5. **Alcance cerrado.** Ver sección 8 (Non-Goals). Cualquier tentación de añadir autenticación, base de datos, multiusuario o tiempo real = parar y preguntar.
6. **No modifiques `riskpkg`.** Si crees que el paquete necesita un cambio (p. ej. exponer una función que hoy es privada), proponlo por separado; no lo edites sobre la marcha mezclando capas.

---

## 3. Contexto del backend (`riskpkg`)

Paquete Python instalable (`pip install -e .`, `pyproject.toml`) organizado en siete subpaquetes que reflejan la arquitectura por capas. **Verifica los nombres exactos al explorar**, esta es la estructura documentada:

| Subpaquete | Responsabilidad |
| --- | --- |
| `riskpkg.data` | Acceso a datos (yfinance), retornos, limpieza, validación |
| `riskpkg.metrics` | Núcleo cuantitativo: VaR, ES, Sharpe, Sortino, drawdown, benchmark, atribución (MRC/PRC), Kupiec |
| `riskpkg.models` | IA/simulación: GARCH/GJR-GARCH, Isolation Forest, Random Forest, Monte Carlo |
| `riskpkg.stress` | Stress testing histórico / hipotético / reverse |
| `riskpkg.levels` | Orquestación de los 4 niveles funcionales (analizadores) |
| `riskpkg.viz` | Presentación: figuras matplotlib |
| `riskpkg.utils` | Constantes (`RANDOM_SEED`) y configuración transversal |

Fachadas de compatibilidad documentadas: `RiskMetrics` (métricas) y `AI_ModelingLayer` (modelado). Analizadores por nivel del estilo `Level3_PortfolioAnalyzer` (verifica los nombres reales de los cuatro). Submódulos de stress documentados: histórico (catálogo de 8 ventanas), hipotético (catálogo regulatorio EBA/CCAR + `FactorShock` configurable), reverse (solución Mahalanobis-óptima + `reverse_stress_curve`).

**Notebooks de demostración existentes** (úsalos como referencia de uso correcto de la API y como oráculo de correctitud): `01_demo_completo.py` y `06_stress_testing.py`.

---

## 4. Stack tecnológico

- **Streamlit** con multipágina nativa (carpeta `pages/`).
- **matplotlib** vía `st.pyplot()` (NO Plotly en el MVP).
- **Python 3.11**.
- `riskpkg` instalado en modo editable en el mismo entorno.
- Sin base de datos, sin backend adicional, sin frontend JS/React.

---

## 5. Arquitectura de la aplicación

- **Sidebar global = constructor de cartera único.** La cartera (universo de activos, pesos, rango de fechas, benchmark, tasa libre de riesgo, fuente de datos) se define **una sola vez** en la barra lateral y se guarda en `st.session_state`. Cada página lee de ahí; **nunca** se vuelve a pedir la cartera en cada pestaña.
- **Una página por nivel** + una página de stress + una home. El mapeo página↔sección del documento debe ser explícito (la home lo indica) para facilitar la defensa.
- **Caching agresivo.** `@st.cache_data` sobre descargas de datos y cómputos caros (ajuste GARCH, Monte Carlo). El determinismo de `riskpkg` hace que el cacheo sea seguro.
- **Sin persistencia entre sesiones.** Cada arranque parte de cero. Solo `session_state`.

### Esquema de `session_state`

Define un único objeto de configuración de cartera (dataclass o `TypedDict`) y guárdalo bajo una clave fija. Campos mínimos:

```python
# components/state.py  (orientativo — ajusta a la API real de riskpkg)
@dataclass
class PortfolioConfig:
    tickers: list[str]
    weights: list[float]            # debe sumar 1.0; validar y renormalizar
    start_date: date
    end_date: date
    benchmark: str                  # p.ej. "^GSPC"
    risk_free_rate: float           # anualizada
    data_source: str                # "demo" | "live"
    # Solo Nivel 4: activos no financieros introducidos a mano
    non_financial_assets: list[dict] = field(default_factory=list)
    # cada dict: {name, value, asset_class, liquidity, est_volatility, corr_market}
```

Una función `init_state()` inicializa el estado vacío; `get_config()` / `set_config()` encapsulan el acceso. Las páginas que requieren cartera deben comprobar que existe y, si no, mostrar un aviso amable que invite a configurarla en el sidebar (no romper con un traceback).

---

## 6. Estructura de ficheros propuesta

```
web/
  app.py                       # Home: explica el sistema, mapea páginas↔secciones del TFG
  pages/
    1_Nivel_1_Activo.py
    2_Nivel_2_Fondo.py
    3_Nivel_3_Cartera.py
    4_Nivel_4_Patrimonio.py
    5_Stress_Testing.py
  components/
    state.py                   # init/get/set de session_state (PortfolioConfig)
    sidebar.py                 # render del constructor de cartera + validaciones
    cache.py                   # wrappers @st.cache_data sobre riskpkg.data y cómputos caros
    formatting.py              # helpers de formato (%, moneda €, tablas df→st)
  data_cache/                  # series pre-descargadas para modo demo (parquet)
docs/
  WEB_SPEC.md                  # este documento
  API_MAP.md                   # lo generas tú tras explorar riskpkg
```

Las **páginas no contienen lógica**: orquestan `components` + `riskpkg` y renderizan. La lógica reutilizable (validación de pesos, formato, wrappers de caché) vive en `components`.

---

## 7. Detalle por página

Para cada página de nivel: leer `PortfolioConfig` del estado → instanciar el analizador de `riskpkg.levels` correspondiente → mostrar (a) tabla de métricas con formato, (b) figuras de `riskpkg.viz` vía `st.pyplot()`. Las métricas concretas de cada nivel son las definidas en el documento (sección 5.4) — **toma la lista real del `API_MAP.md`**, no la dupliques aquí.

- **Nivel 1 — Activo.** Selector del activo concreto dentro del universo. Métricas básicas + comparación volatilidad histórica vs GARCH. Si GARCH no converge, mostrar claramente que se ha usado el **fallback a volatilidad histórica** (no ocultarlo: es un punto fuerte de robustez para la defensa).
- **Nivel 2 — Fondo.** Métricas ponderadas, varianza matricial, ratio de diversificación, **MRC/PRC** por componente (tabla + heatmap de contribución si existe en `viz`).
- **Nivel 3 — Cartera.** Análisis vs benchmark (alpha, beta, R², tracking error, IR, capture ratios), Monte Carlo (bandas de percentiles), Isolation Forest (anomalías sobre serie) y Random Forest (importancia de factores + correlación Spearman con MRC, criterio ρ > 0,70). Es la página más completa; trátala como vitrina.
- **Nivel 4 — Patrimonio.** **Formulario** para añadir activos no financieros (nombre, valor, clase, liquidez, volatilidad estimada, correlación con mercado) que se acumulan en `session_state.non_financial_assets`. Consolidación + descomposición de riesgo por clase y liquidez. Mantener el alcance del MVP documentado: parámetros manuales, sin valoración inmobiliaria automática.

### Página de Stress Testing (vitrina para la defensa)

Tres pestañas (`st.tabs`):

1. **Histórico.** Selector de escenario (8 ventanas: dot-com, GFC, Eurozona, China, Volmageddon, COVID, Inflación 2022, SVB) + botón "batería comparativa" que ejecuta los 8 y muestra el cuadro consolidado y la senda de estrés.
2. **Hipotético.** Catálogo regulatorio (EBA Adverse 2023, CCAR Severely Adverse 2024, estanflación, +200 pb) + editor de `FactorShock` personalizado (shock por clase de activo y override por ticker).
3. **Reverse.** Pérdida objetivo (slider 5%–50%) → shock Mahalanobis-óptimo + **curva de plausibilidad** (`reverse_stress_curve`) con el punto de inflexión. Indicar la interpretación (distancia en σ).

---

## 8. Non-Goals (qué NO construir)

Construir cualquiera de esto significa que te has salido del alcance. **Para y pregunta** si crees que hace falta:

- Autenticación, usuarios, roles, login.
- Base de datos o persistencia de carteras entre sesiones.
- Datos de mercado en tiempo real / live feeds.
- Optimización de carteras (Markowitz, Black-Litterman, risk parity) — el sistema **analiza** riesgo, no lo optimiza.
- Reinforcement Learning, ejecución de órdenes, trading.
- API REST, despliegue SaaS, contenedores de producción.
- Carteras con >100 activos o series intradiarias.
- Reescribir visualizaciones que ya existen en `riskpkg.viz`.

---

## 9. Fases de desarrollo

Trabaja **fase a fase**, haz commit al cerrar cada una, resume lo hecho y **para para que lo revise** antes de seguir. No empieces la fase siguiente sin visto bueno.

### Fase 0 — Andamiaje + cartera en sesión
- Estructura de ficheros, `app.py` home, esqueleto de las 5 páginas (aunque vacías).
- `components/state.py` con `PortfolioConfig`, `init_state`, `get/set`.
- `components/sidebar.py`: constructor de cartera completo con validación (pesos suman 1, fechas coherentes, mínimo de activos) y selector de fuente `demo`/`live`.
- `components/cache.py`: wrappers `@st.cache_data` sobre la descarga de datos de `riskpkg.data`.
- **Modo demo**: script que pre-descarga las series de los casos de prueba a `data_cache/*.parquet`, y carga desde ahí cuando `data_source == "demo"`.
- **Verificación:** `streamlit run web/Home.py` arranca; se define una cartera en el sidebar y el estado persiste al navegar entre páginas; el modo demo carga sin tocar la red.

### Fase 1 — Niveles 1, 2 y 3
- Implementar las tres páginas consumiendo los analizadores reales de `riskpkg.levels` y las figuras de `riskpkg.viz`.
- **Verificación (la importante):** para los inputs de los casos de prueba del documento (Apple 5 años; 60/40 SPY/TLT; cartera internacional vs S&P 500), los números mostrados en la web **coinciden con los de los notebooks** `01_demo_completo.py`. Cualquier discrepancia es un bug y debe investigarse antes de continuar.

### Fase 2 — Stress testing + Nivel 4
- Página de stress con las tres pestañas (sección 7) y página de Nivel 4 con su formulario.
- **Verificación:** reproduce las salidas de `06_stress_testing.py` para los mismos parámetros.

### Fase 3 — Pulido y robustez
- Manejo de errores y casos límite (sección 10), estados de carga (`st.spinner`), mensajes claros.
- Repaso de copy/labels en español, coherencia visual, mapeo página↔sección visible en la home.
- Reservar margen para los días de testing manual de extremo a extremo.

---

## 10. Manejo de errores y casos límite

La web nunca debe romper con un traceback delante del tribunal. Captura y muestra mensajes claros para, al menos:

- **Ticker inválido / fallo de yfinance.** Mensaje accionable + sugerir modo demo.
- **Menos de 252 observaciones.** `riskpkg` ya valida este mínimo; captura el error y explica que el período es insuficiente para métricas fiables.
- **GARCH no converge.** Mostrar explícitamente que se aplicó el fallback a volatilidad histórica (es una feature, no un fallo).
- **Activo inexistente en la fecha de un escenario de stress.** `riskpkg` ya lo excluye y renormaliza pesos; informa de ello en la UI.
- **Pesos que no suman 1.** Validar en el sidebar antes de permitir el análisis (renormalizar con aviso, o bloquear).

---

## 11. Convenciones de código

- Type hints y docstrings en todas las funciones de `components`.
- **Código en inglés, etiquetas de UI en español** (coherente con la memoria del TFG).
- Páginas finas: nada de lógica de cálculo en `pages/`.
- Funciones puras y testeables en `components` (un cálculo de cartera = una función).
- Sin estado global mutable fuera de `session_state`.
- Comentar **por qué**, no el qué obvio.

---

## 12. Cómo reportar el progreso

Al cerrar cada fase, entrega: (1) resumen de qué se implementó, (2) cómo verificarlo en local (`streamlit run ...` + pasos), (3) lista de cualquier hueco encontrado en la API de `riskpkg` (funciones que faltan o que tuviste que sortear), (4) decisiones que tomaste y que convendría que yo confirme. Después, **para**.
