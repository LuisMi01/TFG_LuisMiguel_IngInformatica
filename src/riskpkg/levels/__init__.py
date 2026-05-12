# -*- coding: utf-8 -*-
"""
Orquestación funcional por niveles de agregación patrimonial.

  Nivel 1 → Activo individual
  Nivel 2 → Fondo de inversión
  Nivel 3 → Cartera diversificada + Benchmark + IA + Monte Carlo
  Nivel 4 → Patrimonio global (activos financieros + no financieros)
"""
from .level1_asset import Level1_AssetAnalyzer
from .level2_fund import Level2_FundAnalyzer
from .level3_portfolio import Level3_PortfolioAnalyzer
from .level4_patrimony import Level4_PatrimonyAnalyzer

__all__ = [
    "Level1_AssetAnalyzer",
    "Level2_FundAnalyzer",
    "Level3_PortfolioAnalyzer",
    "Level4_PatrimonyAnalyzer",
]
