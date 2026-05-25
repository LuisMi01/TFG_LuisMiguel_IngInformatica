# -*- coding: utf-8 -*-
"""
Stress testing hipotético: aplicación de shocks definidos por el analista
sobre la cartera, sin depender de datos históricos.

Soporta dos niveles de granularidad:
    - Shocks por activo individual (override puntual)
    - Shocks por clase de activo (renta variable, renta fija, materias primas...)

Incluye un catálogo de escenarios regulatorios predefinidos inspirados en
las metodologías de la Autoridad Bancaria Europea (EBA) y la Reserva Federal
estadounidense (CCAR — Comprehensive Capital Analysis and Review).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class FactorShock:
    """
    Definición de un shock instantáneo a aplicar sobre una cartera.

    El shock se especifica primero por clase de activo (afectando a todos los
    activos de esa clase) y, opcionalmente, se permiten overrides por ticker
    individual. Los activos sin clase ni override reciben `default_shock`.

    Ejemplo:
        eba_adverse = FactorShock(
            name="EBA Adverso 2023",
            class_shocks={"equity": -0.30, "fixed_income": -0.10, "commodity": +0.05},
            default_shock=0.0,
        )

    Notación: shocks expresados como variación porcentual (-0.30 = caída del 30%).
    """
    name: str
    class_shocks: Dict[str, float] = field(default_factory=dict)
    ticker_overrides: Dict[str, float] = field(default_factory=dict)
    default_shock: float = 0.0
    description: str = ""


# ─── Catálogo de escenarios regulatorios predefinidos ───────────────────────
PREDEFINED_SHOCKS: Dict[str, FactorShock] = {

    "eba_adverse_2023": FactorShock(
        name="EBA Escenario Adverso 2023",
        class_shocks={
            "equity":       -0.30,
            "fixed_income": -0.10,
            "real_estate":  -0.20,
            "commodity":    +0.05,
            "cash":         0.0,
            "alternative":  -0.15,
        },
        description=(
            "Escenario adverso del stress test bancario de la EBA (2023): "
            "recesión severa simultánea en la UE, caída de bolsas del 30% y "
            "subida de spreads de crédito."
        ),
    ),

    "ccar_severely_adverse_2024": FactorShock(
        name="CCAR Severely Adverse 2024",
        class_shocks={
            "equity":       -0.55,
            "fixed_income": -0.05,
            "real_estate":  -0.40,
            "commodity":    -0.10,
            "cash":         0.0,
            "alternative":  -0.30,
        },
        description=(
            "Escenario severamente adverso de la Fed (CCAR 2024): recesión "
            "global profunda con caída del 55% en bolsas y crash inmobiliario."
        ),
    ),

    "stagflation": FactorShock(
        name="Estanflación (1970s revisited)",
        class_shocks={
            "equity":       -0.25,
            "fixed_income": -0.15,
            "real_estate":  -0.10,
            "commodity":    +0.40,
            "cash":         -0.08,  # erosión real por inflación
            "alternative":  -0.10,
        },
        description=(
            "Inflación persistente y crecimiento débil. Penaliza bonos y "
            "acciones simultáneamente y favorece materias primas."
        ),
    ),

    "rate_shock_200bp": FactorShock(
        name="Subida de tipos +200 bp",
        class_shocks={
            "equity":       -0.15,
            "fixed_income": -0.12,  # duración media 6 años aprox.
            "real_estate":  -0.18,
            "commodity":    -0.05,
            "cash":         +0.02,
            "alternative":  -0.10,
        },
        description=(
            "Subida abrupta de 200 puntos básicos en tipos de referencia. "
            "Penaliza activos de larga duración (renta fija e inmobiliario)."
        ),
    ),
}


def apply_shock(
    weights: Union[List[float], np.ndarray],
    tickers: List[str],
    shock: Union[str, FactorShock],
    ticker_to_class: Optional[Dict[str, str]] = None,
    initial_value: float = 10_000.0,
) -> Dict:
    """
    Aplica un shock hipotético a una cartera y calcula la pérdida resultante.

    Parameters
    ----------
    weights : array-like
        Pesos actuales de la cartera (suman 1).
    tickers : List[str]
        Lista de tickers en el mismo orden que `weights`.
    shock : str | FactorShock
        Clave de PREDEFINED_SHOCKS o instancia de FactorShock.
    ticker_to_class : Dict[str, str], opcional
        Mapeo ticker → clase de activo. Necesario si el shock se define por
        clase de activo y no por ticker.
    initial_value : float
        Valor de partida de la cartera.

    Returns
    -------
    dict con:
        - shock_name, shock_description : metadatos del escenario
        - asset_shocks                  : dict ticker -> shock aplicado (%)
        - asset_contributions           : dict ticker -> contribución a la pérdida (€)
        - portfolio_pnl_pct             : pérdida total como % de la cartera
        - portfolio_value_after         : valor de la cartera tras el shock
    """
    # Resolver shock
    if isinstance(shock, str):
        if shock not in PREDEFINED_SHOCKS:
            available = ", ".join(PREDEFINED_SHOCKS.keys())
            raise KeyError(f"Shock '{shock}' no encontrado. Disponibles: {available}")
        shk = PREDEFINED_SHOCKS[shock]
    else:
        shk = shock

    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Los pesos deben sumar 1. Suma actual: {weights.sum():.6f}")
    if len(weights) != len(tickers):
        raise ValueError("weights y tickers deben tener la misma longitud.")

    ticker_to_class = ticker_to_class or {}

    # Resolver shock por activo: prioridad override > clase > default
    asset_shocks = {}
    for t in tickers:
        if t in shk.ticker_overrides:
            asset_shocks[t] = shk.ticker_overrides[t]
        elif t in ticker_to_class and ticker_to_class[t] in shk.class_shocks:
            asset_shocks[t] = shk.class_shocks[ticker_to_class[t]]
        else:
            asset_shocks[t] = shk.default_shock

    # Vector de shocks alineado con los pesos
    shock_vec = np.array([asset_shocks[t] for t in tickers])

    # P&L de la cartera (asume rebalanceo instantáneo)
    portfolio_pnl_pct = float(np.dot(weights, shock_vec))
    portfolio_value_after = initial_value * (1 + portfolio_pnl_pct)

    # Contribución en euros de cada activo
    asset_contributions = {
        t: float(weights[i] * shock_vec[i] * initial_value)
        for i, t in enumerate(tickers)
    }

    return {
        "shock_name":             shk.name,
        "shock_description":      shk.description,
        "asset_shocks":           asset_shocks,
        "asset_contributions":    asset_contributions,
        "ticker_to_class":        ticker_to_class,
        "initial_value":          initial_value,
        "portfolio_value_after":  portfolio_value_after,
        "portfolio_pnl_pct":      portfolio_pnl_pct,
        "portfolio_pnl_eur":      portfolio_value_after - initial_value,
    }


def list_predefined_shocks() -> None:
    """Imprime el catálogo de escenarios hipotéticos predefinidos."""
    print(f"\n{'═'*70}")
    print("  Catálogo de Shocks Hipotéticos Predefinidos")
    print(f"{'═'*70}")
    for s in PREDEFINED_SHOCKS.values():
        print(f"  [{s.name}]")
        for cls, sh in s.class_shocks.items():
            print(f"     {cls:<15} → {sh:>+.1%}")
        print(f"     {s.description}")
        print()
    print(f"{'═'*70}\n")
