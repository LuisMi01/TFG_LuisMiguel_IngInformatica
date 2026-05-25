# -*- coding: utf-8 -*-
"""
riskpkg — Sistema de Análisis y Gestión del Riesgo Financiero en
Carteras Multi-Activo mediante Técnicas Cuantitativas e Inteligencia Artificial.

Trabajo de Fin de Grado — Grado en Ingeniería Informática
Autor: Luis Miguel Urbez Villar
Curso académico: 2025–2026

Arquitectura por capas:
    riskpkg.data     → Capa de acceso a datos
    riskpkg.metrics  → Capa de lógica de negocio (métricas cuantitativas)
    riskpkg.models   → Capa de modelado (IA + simulación)
    riskpkg.levels   → Orquestación funcional por niveles (1–4)
    riskpkg.viz      → Capa de presentación (visualizaciones)
    riskpkg.utils    → Constantes y utilidades transversales

Quickstart:
    >>> from riskpkg.levels import Level3_PortfolioAnalyzer
    >>> from riskpkg.viz import RiskVisualizer
    >>> portfolio = Level3_PortfolioAnalyzer(
    ...     tickers=["AAPL", "MSFT", "GLD"],
    ...     weights=[0.4, 0.4, 0.2],
    ...     start_date="2020-01-01", end_date="2024-12-31",
    ... ).run()
    >>> portfolio.print_report()
"""

__version__ = "0.5.0"  # 0.5 = añade EVT-POT (GPD) + suite pytest + CI

# ── Re-exports de alto nivel para uso conveniente ──────────────────────────
from .data import DataLoader
from .metrics import RiskMetrics
from .models import AI_ModelingLayer
from .levels import (
    Level1_AssetAnalyzer,
    Level2_FundAnalyzer,
    Level3_PortfolioAnalyzer,
    Level4_PatrimonyAnalyzer,
)
from .viz import RiskVisualizer
from .utils.constants import (
    TRADING_DAYS_YEAR,
    DEFAULT_CONFIDENCE,
    DEFAULT_RISK_FREE,
    BENCHMARK_TICKER,
    RANDOM_SEED,
)

__all__ = [
    "__version__",
    # Capas
    "DataLoader",
    "RiskMetrics",
    "AI_ModelingLayer",
    "RiskVisualizer",
    # Niveles
    "Level1_AssetAnalyzer",
    "Level2_FundAnalyzer",
    "Level3_PortfolioAnalyzer",
    "Level4_PatrimonyAnalyzer",
    # Constantes
    "TRADING_DAYS_YEAR",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RISK_FREE",
    "BENCHMARK_TICKER",
    "RANDOM_SEED",
]
