# -*- coding: utf-8 -*-
"""
Configuración global del sistema de análisis de riesgo.

Todos los parámetros que pueden ser ajustados por el usuario están
centralizados aquí. La semilla aleatoria global se fija al importar
este módulo para garantizar reproducibilidad.
"""
import numpy as np

# ─── Constantes de mercado ──────────────────────────────────────────────────
TRADING_DAYS_YEAR   = 252                # Días de negociación / año (estándar US/EU)
DEFAULT_CONFIDENCE  = 0.95               # Nivel de confianza por defecto para VaR/ES
DEFAULT_RISK_FREE   = 0.0361             # Tipo libre de riesgo (BCE 2024: 3,61% anual)

# ─── Configuración de datos ─────────────────────────────────────────────────
DEFAULT_START_DATE  = "2015-01-01"
DEFAULT_END_DATE    = "2025-12-31"
BENCHMARK_TICKER    = "SPY"              # Benchmark por defecto (S&P 500 ETF)
MIN_OBSERVATIONS    = 252                # Mínimo de sesiones para métricas fiables

# ─── Reproducibilidad ───────────────────────────────────────────────────────
RANDOM_SEED         = 42                 # Semilla global

# Aplicación inmediata de la semilla al importar el paquete.
# Cualquier función que use np.random posteriormente parte de un estado conocido.
np.random.seed(RANDOM_SEED)
