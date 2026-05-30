# API_MAP.md — Contrato real de `riskpkg` para la capa web

> Documento generado tras explorar el código fuente de `riskpkg` (v0.5.0) leyendo
> los módulos uno a uno. **Es la fuente de verdad de firmas y tipos de retorno**
> que consumirá la capa Streamlit. Si una firma cambia en `riskpkg`, actualiza
> primero este fichero. Las referencias de API del `WEB_SPEC.md` son orientativas;
> **esto es lo que el código realmente expone**.
>
> Convención de lectura: `→` indica el tipo de retorno. Los atributos con guion
> bajo (`_attr`) son privados pero **se leen directamente en los notebooks de
> demostración**, así que son el patrón establecido de acceso (ver §8 Huecos).

---

## 0. Realidad del entorno (importante para Fase 0)

- **`riskpkg` NO está instalado en este equipo.** El `.venv/` del repo es un
  entorno **de Windows** (`pyvenv.cfg` apunta a `C:\ProgramData\anaconda3`,
  Python 3.12.4) y no es ejecutable en Linux. El Python del sistema es **3.14.4**
  (el `pyproject.toml` pide `>=3.11`; el TFG fija 3.11).
- Por tanto, antes de cualquier verificación de Fase 0 hace falta crear un venv
  Linux nuevo e instalar `pip install -e ".[dev,notebooks]"` + Streamlit.
- `RANDOM_SEED = 42` se fija en el **import** de `riskpkg.utils.constants`
  (`np.random.seed(42)` a nivel de módulo). Monte Carlo e Isolation Forest
  además fijan la semilla internamente en cada llamada → reproducibilidad bit a bit.

## 1. Estructura y re-exports de alto nivel

`import riskpkg` reexporta (desde `src/riskpkg/__init__.py`):

```python
from riskpkg import (
    DataLoader, RiskMetrics, AI_ModelingLayer, RiskVisualizer,
    Level1_AssetAnalyzer, Level2_FundAnalyzer,
    Level3_PortfolioAnalyzer, Level4_PatrimonyAnalyzer,
    TRADING_DAYS_YEAR, DEFAULT_CONFIDENCE, DEFAULT_RISK_FREE,
    BENCHMARK_TICKER, RANDOM_SEED,
)
```

Constantes (en `riskpkg.utils.constants`):

| Constante | Valor |
|---|---|
| `TRADING_DAYS_YEAR` | `252` |
| `DEFAULT_CONFIDENCE` | `0.95` |
| `DEFAULT_RISK_FREE` | `0.0361` (BCE 2024, anualizado) |
| `DEFAULT_START_DATE` / `DEFAULT_END_DATE` | `"2015-01-01"` / `"2025-12-31"` |
| `BENCHMARK_TICKER` | `"SPY"` (¡no `^GSPC`! usar `SPY` para casar con notebooks) |
| `MIN_OBSERVATIONS` | `252` (solo emite **aviso por print**, no lanza excepción) |
| `RANDOM_SEED` | `42` |

**Cartera demo canónica** (en `config/default.yaml` y en ambos notebooks):

```python
TICKERS = ["ITX.MC", "AMZN", "TTWO", "ANA", "GLD"]
WEIGHTS = [0.30, 0.25, 0.15, 0.20, 0.10]
START, END = "2018-01-01", "2024-12-31"
BENCHMARK = "SPY"; return_type = "log"; mc_sims = 1000; mc_days = 252
# Para stress: INITIAL_VALUE = 100_000.0
TICKER_TO_CLASS = {"ITX.MC":"equity","AMZN":"equity","TTWO":"equity",
                   "ANA":"equity","GLD":"commodity"}
```

---

## 2. `riskpkg.data` — acceso a datos

### `DataLoader(tickers, start_date="2015-01-01", end_date="2025-12-31", return_type="log")`
Métodos / propiedades:
- `.fetch() → DataLoader` — descarga vía `yfinance.download(...)`. **Imprime** una
  línea de log. Encadenable.
- `.load_csv(path) → DataLoader` — alternativa desde CSV (índice de fecha en col 0).
- `.prices → pd.DataFrame` (precios de cierre ajustados).
- `.log_returns → pd.DataFrame`, `.simple_returns → pd.DataFrame`.
- `.returns → pd.DataFrame` — log o simple según `return_type`. **Columnas = tickers.**
- `.summary() → None` (print).

Validación (`_validate`): elimina tickers sin datos (con aviso), hace ffill+dropna,
y si `len < MIN_OBSERVATIONS` **solo imprime un aviso** (no excepción). Solo lanza
`ValueError` si el DataFrame queda vacío (tickers todos inválidos / sin conexión).

> **Clave para "modo demo":** los analizadores de nivel construyen su **propio**
> `DataLoader(...).fetch()` internamente (ver §5). No aceptan datos inyectados.
> Para evitar la red en modo demo hay que interceptar `yfinance.download`
> (ver §8, decisión D1).

---

## 3. `riskpkg.metrics` — núcleo cuantitativo

Funciones puras + fachada `RiskMetrics` (cada método es `staticmethod` de la función pura).

### El dict que lo gobierna todo: `full_report(returns, label="Activo", confidence=0.95, risk_free=0.0361) → Dict`
Claves exactas (todas presentes siempre):
```
label, annual_return, volatility_ann,
var_param_95, var_param_99, var_hist_95, var_hist_99,
expected_shortfall, sharpe_ratio, sortino_ratio,
max_drawdown, recovery_days
```
- `recovery_days` puede ser `None` (no recuperado dentro de la serie).
- VaR/ES devueltos como **pérdidas positivas** (convención banca).

### Funciones sueltas (todas en `RiskMetrics.<nombre>` también)
| Función | Firma → retorno |
|---|---|
| `volatility(returns)` | → `float` (anualizada) |
| `var_parametric(returns, confidence=0.95)` | → `float` |
| `var_historical(returns, confidence=0.95)` | → `float` |
| `expected_shortfall(returns, confidence=0.95)` | → `float` |
| `sharpe_ratio(returns, risk_free=0.0361)` | → `float` |
| `sortino_ratio(returns, risk_free=0.0361)` | → `float` |
| `max_drawdown(returns)` | → `float` (negativo) |
| `drawdown_recovery_period(returns)` | → `int \| None` |
| `kupiec_test(returns, confidence=0.95, method="historical")` | → `Dict` con `interpretation`, `exceedances`, `n_test`, `p_value`, … |
| `alpha_beta(port, bench)` | → `Tuple[float, float]` (alpha anual, beta) |
| `tracking_error(port, bench)` | → `float` |
| `information_ratio(port, bench)` | → `float` |
| `up_down_capture(port, bench)` | → `Tuple[float, float]` (up%, down%) |
| `r_squared(port, bench)` | → `float` |
| `diversification_ratio(asset_returns_df, weights)` | → `float` |
| `diversification_benefit(asset_returns_df, weights)` | → `float` |
| `marginal_risk_contribution(asset_returns_df, weights)` | → `pd.Series` (indexada por ticker) |
| `percentage_risk_contribution(asset_returns_df, weights)` | → `pd.Series` (% del riesgo total) |

Flag: `riskpkg.metrics.HAS_STATSMODELS: bool`.

---

## 4. `riskpkg.models` — IA / simulación

Funciones puras + fachada `AI_ModelingLayer`. Flag `riskpkg.models.HAS_ARCH: bool`.

### `garch_volatility(returns, model_type="GARCH") → Dict`  ( `model_type` ∈ {`"GARCH"`,`"GJR-GARCH"`} )
- **Si converge:** `{model, aic, bic, params(dict), vol_forecast_ann, vol_hist_ann, converged=True}`
- **Si NO converge / sin `arch` / < 252 obs:** `{model:"Fallback ...", vol_forecast_ann, converged=False}`
  → **el fallback a vol histórica es feature, no fallo.** UI debe mostrar `converged`.

### `isolation_forest_anomalies(returns, contamination=0.02, random_state=42) → pd.Series`
Valores `-1` (anomalía) / `1` (normal), índice = fechas.

### `rf_risk_factor_importance(asset_returns_df, portfolio_returns, n_splits=5, random_state=42) → Dict`
`{rf_importances: pd.Series, rf_ranking: list, n_cv_splits: int}` (walk-forward TS-CV).

### `compare_risk_attribution(rf_importances: pd.Series, classical_mrc: pd.Series) → Dict`
`{spearman_correlation, p_value, n_assets, meets_criterion(bool), interpretation(str con ✅/⚠️)}`.
Criterio TFG: ρ > 0.70.

### `monte_carlo_simulation(returns_df, weights, days=252, n_sims=1000, initial_value=10_000, seed=42) → Dict`
Claves: `paths(ndarray n_sims×days)`, `final_values(ndarray)`, `initial_value`,
`percentiles({5,25,50,75,95}→escalar)`, `daily_percentiles({5,25,50,75,95}→ndarray(days,))`,
`mean_final`, `std_final`, `prob_loss(%)`, `var_mc_95`, `days`, `n_sims`, `seed`.

### EVT-POT (en `models`, **no cableado en ningún nivel** — ver decisión D3)
- `mean_excess(returns, thresholds=None, tail="left") → pd.DataFrame`
- `select_threshold(returns, method="percentile", value=0.95, tail="left") → (float, pd.Series)`
- `fit_gpd(excesses) → Dict` (`xi, beta, xi_se, beta_se, n, method`; o `{status,...}` si falla)
- `evt_var_es(returns, confidence=0.95, threshold_pct=0.95, tail="left") → Dict` (`u, xi, beta, var, es, method, …`)
- `evt_report(returns, threshold_pct=0.95, confidences=[.95,.99,.995], tail="left", ad_mc_samples=200) → Dict`
  (incluye `ad_statistic`, `ad_pvalue`, `interpretation`, `var_es` por confianza)
- `compare_var_methods(returns, confidence=0.99, threshold_pct=0.95) → Dict`
  (`var_parametric, var_historical, var_evt, es_evt, ratio_evt_param, xi, interpretation`)

---

## 5. `riskpkg.levels` — analizadores por nivel (lo que cada página instancia)

Patrón común: construir → `.run()` (descarga datos + calcula) → leer atributos.
**Muchos resultados viven en atributos privados** (los notebooks los leen así).

### Nivel 1 — `Level1_AssetAnalyzer(ticker, start_date, end_date, return_type="log")`
`.run()` luego:
- `.metrics → Dict` (= `full_report`, propiedad pública; lanza si no se hizo `run()`)
- `.returns → pd.Series` (propiedad pública)
- `._garch`, `._gjr → Dict` (de `garch_volatility`; **privados**)
- `._kupiec → Dict`
- `.print_report() → None`

**Página Nivel 1 consume:** `metrics` (tabla), `_garch`/`_gjr` (vol histórica vs GARCH vs GJR,
mostrando `converged`), `_kupiec["interpretation"]`. Figura: `RiskVisualizer.plot_var_distribution(returns, label)`.

### Nivel 2 — `Level2_FundAnalyzer(tickers, weights, fund_name="Fondo", start_date, end_date, return_type="log")`
> ⚠️ El `__init__` hace `assert abs(sum(weights)-1.0) < 1e-6` y `len(tickers)==len(weights)`.
> El sidebar debe renormalizar **antes** de instanciar o saltará `AssertionError`.

`.run()` luego (todos **privados** salvo donde se indique):
- `._fund_metrics → Dict` (full_report del fondo)
- `._asset_returns → pd.DataFrame`, `._fund_returns → pd.Series`
- `._asset_metrics → Dict[str, Dict]` (full_report por activo)
- `._mrc → pd.Series`, `._prc → pd.Series`
- `._div_ratio → float`, `._div_benefit → float`
- `.weights_effective → np.ndarray` (pesos renormalizados tras posibles exclusiones)
- `.print_report() → None`

**Página Nivel 2 consume:** `_fund_metrics`, `_div_ratio`, `_div_benefit`, `_mrc`, `_prc`,
`_asset_metrics`. Figuras: `plot_risk_attribution(_mrc, _prc)` y `plot_correlation_matrix(_asset_returns.corr())`.
(No hay heatmap específico de MRC; el de correlación es lo más cercano.)

### Nivel 3 — `Level3_PortfolioAnalyzer(tickers, weights, portfolio_name="Cartera", benchmark="SPY", start_date, end_date, return_type="log", mc_sims=1000, mc_days=252)`
> ⚠️ `assert abs(sum(weights)-1.0) < 1e-6` en `__init__`.

`.run()` luego:
- **Públicos:** `.asset_returns → pd.DataFrame`, `.port_returns → pd.Series`,
  `.bench_returns → pd.Series | None`, `.weights_effective → np.ndarray`.
- **Privados:** `._port_metrics`, `._bench_metrics`, `._corr_matrix → pd.DataFrame`,
  `._mrc`, `._prc → pd.Series`, `._div_ratio → float`,
  `._alpha_beta → (alpha,beta) | None`, `._tracking_error`, `._info_ratio`, `._r2 → float`,
  `._up_down_capture → (up,down)`, `._kupiec → Dict`,
  `._anomalies → pd.Series`, `._rf_results → Dict`, `._spearman → Dict`,
  `._mc_results → Dict`.
- `.print_report() → None`

**Página Nivel 3 consume (la más rica):** `_port_metrics` + benchmark (`_alpha_beta`,
`_r2`, `_tracking_error`, `_info_ratio`, `_up_down_capture`), `_div_ratio`, `_kupiec`,
anomalías, RF + Spearman, Monte Carlo. Figuras disponibles:
`plot_cumulative_returns(asset_returns, benchmark_returns=bench_returns)`,
`plot_correlation_matrix(_corr_matrix)`,
`plot_risk_attribution(_mrc, _prc, rf_importances=_rf_results["rf_importances"])`,
`plot_anomalies(port_returns, _anomalies)`,
`plot_monte_carlo(_mc_results)`,
`plot_rolling_metrics(port_returns, bench_returns)` (solo si hay benchmark).

### Nivel 4 — `Level4_PatrimonyAnalyzer(financial_portfolio: Level3_PortfolioAnalyzer, non_financial_assets: List[Dict]=None, patrimony_name="Patrimonio Global")`
> **Depende de un `Level3_...().run()` ya ejecutado.** No descarga datos propios.
> ⚠️ **`fin_value_ref = 1_000_000` está HARDCODED**: la cartera financiera se valora
> siempre en 1 M€ de referencia, independientemente de los pesos. Los activos no
> financieros se ponderan **relativos a ese 1 M€**. Es una simplificación MVP
> documentada; la UI debe explicitarlo.

Estructura **exacta** de cada dict en `non_financial_assets` (claves reales):
```python
{ "name": str, "class": str, "liquidity": str, "value": float,
  "annual_return_est": float, "volatility_est": float,
  "correlation_with_market": float }
```
- `class` ∈ {`equity`,`fixed_income`,`real_estate`,`commodity`,`alternative`,`cash`}
- `liquidity` ∈ {`liquid`,`semi_liquid`,`illiquid`}
- ⚠️ **Ojo:** el `WEB_SPEC.md` propone los nombres `asset_class`, `est_volatility`,
  `corr_market`. **NO son los reales.** El formulario debe emitir las claves de arriba.

`.run()` rellena `._results → Dict`:
```
patrimony_name, financial_metrics, fin_value_ref, nf_total_value, total_value,
fin_weight, nf_items(list[dict]), var_patrimonial, by_class(dict), by_liquidity(dict)
```
`.print_report() → None`. Figura: `RiskVisualizer.plot_patrimony_breakdown(_results)`.

---

## 6. `riskpkg.viz` — figuras (¡OJO al tipo de retorno!)

`RiskVisualizer` (todos `staticmethod`). **TODAS las funciones devuelven `None` y
llaman a `plt.show()` internamente — NO devuelven el objeto `fig`.**

> Implicación para Streamlit (decisión D2): no se puede hacer `fig = plot_x(...)`.
> El patrón será: backend `Agg`, llamar a la función, y capturar la figura activa:
> ```python
> import matplotlib; matplotlib.use("Agg")
> RiskVisualizer.plot_var_distribution(returns, label="AAPL")
> st.pyplot(plt.gcf()); plt.close("all")
> ```
> Esto **no toca `riskpkg`** (cumple regla #6). Alternativa más limpia (refactor en
> `riskpkg` para devolver `fig`) queda como propuesta separada, no se hace sobre la marcha.

| Función | Argumentos clave |
|---|---|
| `plot_cumulative_returns(returns_df, title=..., benchmark_returns=None)` | df de activos + Series benchmark opcional |
| `plot_var_distribution(returns_series, confidence=0.95, label="Activo")` | Series de un activo/cartera |
| `plot_correlation_matrix(corr_matrix_df, title=...)` | DataFrame de correlación |
| `plot_rolling_metrics(port_returns, bench_returns, window=63, title=...)` | dos Series alineadas |
| `plot_monte_carlo(mc_results: Dict, title=...)` | el dict de `monte_carlo_simulation` |
| `plot_risk_attribution(mrc, prc, rf_importances=None, title=...)` | Series MRC/PRC (+RF opcional) |
| `plot_anomalies(returns, anomalies, title=...)` | Series retornos + Series ±1 |
| `plot_patrimony_breakdown(results: Dict, title=...)` | el `_results` del Nivel 4 |
| `plot_historical_stress_path(result: Dict, title=None)` | dict de `historical_stress_test` |
| `plot_historical_stress_battery(battery: pd.DataFrame, title=...)` | DataFrame de la batería |
| `plot_reverse_stress_curve(curve: pd.DataFrame, title=...)` | DataFrame de `reverse_stress_curve` |

---

## 7. `riskpkg.stress` — stress testing

### Histórico
- `HISTORICAL_SCENARIOS: Dict[str, StressWindow]` — 8 ventanas. Claves exactas:
  `dot_com_2000`, `gfc_2008`, `eurozone_2011`, `china_2015`, `volmageddon_2018`,
  `covid_2020`, `inflation_2022`, `svb_2023`.
- `StressWindow` (dataclass frozen): `key, name, start, end, description`.
- `list_scenarios() → None` (print) · `get_scenario(key) → StressWindow` (KeyError si no existe).
- `historical_stress_test(tickers, weights, scenario, return_type="log", initial_value=10_000.0) → Dict`
  - `scenario` = clave `str` o `StressWindow`. **Descarga datos de la ventana vía yfinance.**
  - Retorno: `scenario, scenario_name, scenario_description, start, end, n_sessions,
    tickers_used, tickers_excluded, weights_effective(dict), initial_value,
    portfolio_value_final, portfolio_value_min, portfolio_pnl_pct, max_drawdown_pct,
    days_to_trough, recovered_within_window(bool), asset_pnl_pct(dict),
    portfolio_path(pd.Series)`.
  - Excluye activos sin datos en la ventana y **renormaliza pesos** (lo informa en `tickers_excluded`).
- `historical_stress_battery(tickers, weights, scenarios=None, return_type="log", initial_value=10_000.0) → pd.DataFrame`
  - Una fila por escenario (índice = `scenario_name`). Columnas: `start, end, n_sessions,
    pnl_pct, max_drawdown_pct, days_to_trough, recovered, value_final, n_assets_excluded`
    (o `error` si ese escenario falló). `scenarios=None` → los 8.

### Hipotético
- `FactorShock` (dataclass frozen): `name, class_shocks: Dict[str,float],
  ticker_overrides: Dict[str,float]={}, default_shock=0.0, description=""`.
  Shocks como variación proporcional (`-0.30` = −30%).
- `PREDEFINED_SHOCKS: Dict[str, FactorShock]` — claves: `eba_adverse_2023`,
  `ccar_severely_adverse_2024`, `stagflation`, `rate_shock_200bp`.
- `apply_shock(weights, tickers, shock, ticker_to_class=None, initial_value=10_000.0) → Dict`
  - `shock` = clave `str` o `FactorShock`. **`ticker_to_class` es necesario** para que
    los shocks por clase apliquen (prioridad: override por ticker > clase > `default_shock`).
  - **No descarga datos** (cálculo instantáneo).
  - Retorno: `shock_name, shock_description, asset_shocks(dict), asset_contributions(dict €),
    ticker_to_class, initial_value, portfolio_value_after, portfolio_pnl_pct, portfolio_pnl_eur`.
- `list_predefined_shocks() → None` (print).

### Reverse
- `reverse_stress_test(asset_returns_df, weights, target_loss, horizon="daily") → Dict`
  - `horizon` ∈ {`"daily"`,`"annual"`}; `target_loss` > 0 (proporción, p.ej. 0.20).
  - Usa `asset_returns` (típicamente `lvl3.asset_returns`) y `lvl3.weights_effective`.
  - Retorno: `target_loss, horizon, shock_vector(pd.Series), portfolio_loss_check,
    mahalanobis_distance, plausibility_zscore, plausibility_prob, sigma_portfolio`.
- `reverse_stress_curve(asset_returns_df, weights, losses=None, horizon="daily") → pd.DataFrame`
  - `losses` por defecto `[0.05..0.50]`. Índice = `target_loss_pct`; columnas:
    `mahalanobis_distance, plausibility_prob, sigma_portfolio`.

### Reporters (todos print, devuelven `None`)
`print_historical_report(dict)`, `print_hypothetical_report(dict)`,
`print_reverse_report(dict)`, `print_historical_battery_report(df)`.
→ La web **no** los usará (reconstruirá tablas con `st`), pero sirven de oráculo de formato.

---

## 8. Huecos en la API y decisiones a confirmar

| # | Hueco / hallazgo | Impacto en la web | Propuesta |
|---|---|---|---|
| **D1** | Los analizadores de nivel y el stress histórico construyen su **propio** `DataLoader(...).fetch()` (yfinance). No aceptan datos inyectados. | El "modo demo offline" del WEB_SPEC no es trivial: no basta con cachear `riskpkg.data`. | **Interceptar `yfinance.download`** en el arranque del proceso Streamlit, devolviendo datos desde `data_cache/*.parquet` para los tickers conocidos. No toca `riskpkg` (regla #6). Cachear además el **resultado completo del analizador** con `@st.cache_data`. |
| **D2** | Las funciones de `viz` hacen `plt.show()` y devuelven `None` (no `fig`). | `st.pyplot(fig)` no recibe `fig`. | Backend `Agg` + `st.pyplot(plt.gcf())` + `plt.close("all")`. Funciona sin tocar `riskpkg`. |
| **D3** | EVT-POT (`compare_var_methods`, `evt_report`) existe en `models` pero **ningún nivel lo cablea** y el WEB_SPEC no le da página. | Quedaría una capacidad potente sin vitrina (leptocurtosis es punto fuerte del TFG). | Confirmar si añadir un panel EVT en la página de Nivel 1 (o Nivel 3). Sin tocar `riskpkg` (las funciones ya son públicas). |
| **D4** | Resultados de niveles 2/3 viven en **atributos privados** (`_mrc`, `_rf_results`, ...). | La web los leerá igual que los notebooks. Frágil si `riskpkg` cambia. | Aceptable para MVP (es el patrón del propio repo). Opcional: proponer accesores públicos en `riskpkg` como cambio separado. |
| **D5** | Nivel 4 fija `fin_value_ref = 1_000_000 €` hardcoded; las claves del dict no financiero difieren de las del WEB_SPEC. | El formulario debe usar las claves reales (§5) y la UI debe explicar el supuesto de 1 M€. | Implementar con claves reales; texto aclaratorio en la página. |
| **D6** | `DataLoader` **no lanza excepción** con < 252 obs (solo `print`). Solo lanza `ValueError` si no hay datos en absoluto. | El manejo de "< 252 obs" del WEB_SPEC §10 no puede basarse en `try/except`. | La web comprobará `len(returns)` por su cuenta (o el nº de sesiones devuelto) y mostrará el aviso. |
| **D7** | Casos de prueba del WEB_SPEC ("Apple 5y", "60/40 SPY/TLT", "internacional vs S&P500") **no coinciden** con los notebooks reales (cartera `ITX.MC/AMZN/TTWO/ANA/GLD`, benchmark `SPY`). | El oráculo de correctitud de Fase 1/2 son los **notebooks**, no esos casos. | Usar la cartera demo de `config/default.yaml` como oráculo. Confirmar. |
| **D8** | `data_cache/` y `*.parquet` están en `.gitignore`. | Tras un `git clone` limpio el modo demo no tendría datos → riesgo el día de la defensa. | Decidir: (a) **des-ignorar** los parquet de demo y commitearlos, o (b) ejecutar el script de pre-descarga antes de la defensa. |

---

*Generado en Fase 0 (exploración). Actualizar si cambia la API de `riskpkg`.*
