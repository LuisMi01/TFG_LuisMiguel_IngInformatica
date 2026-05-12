# -*- coding: utf-8 -*-
"""
Capa de stress testing.

Tres aproximaciones complementarias:

  · stress.historical    — Reaplicación de ventanas reales de crisis pasadas
                           (GFC 2008, COVID 2020, etc.) a la cartera actual.
  · stress.hypothetical  — Shocks instantáneos definidos por el analista o
                           tomados de catálogos regulatorios (EBA, CCAR).
  · stress.reverse       — Búsqueda inversa: dada una pérdida objetivo,
                           identifica el escenario más plausible que la causa.

Referencias:
    Breuer, T. & Csiszár, I. (2013). "Systematic stress tests with entropic
        plausibility constraints", Journal of Banking & Finance 37(5).
    EBA (2023). "2023 EU-Wide Stress Test Methodological Note".
    Federal Reserve (2024). "Dodd-Frank Act Stress Test 2024: Supervisory
        Stress Test Methodology".
"""
from .scenarios import (
    StressWindow,
    HISTORICAL_SCENARIOS,
    list_scenarios,
    get_scenario,
)
from .historical import (
    historical_stress_test,
    historical_stress_battery,
)
from .hypothetical import (
    FactorShock,
    PREDEFINED_SHOCKS,
    apply_shock,
    list_predefined_shocks,
)
from .reverse import (
    reverse_stress_test,
    reverse_stress_curve,
)
from .reporter import (
    print_historical_report,
    print_hypothetical_report,
    print_reverse_report,
    print_historical_battery_report,
)


__all__ = [
    # Escenarios
    "StressWindow",
    "HISTORICAL_SCENARIOS",
    "list_scenarios",
    "get_scenario",
    # Histórico
    "historical_stress_test",
    "historical_stress_battery",
    # Hipotético
    "FactorShock",
    "PREDEFINED_SHOCKS",
    "apply_shock",
    "list_predefined_shocks",
    # Reverse
    "reverse_stress_test",
    "reverse_stress_curve",
    # Reporter
    "print_historical_report",
    "print_hypothetical_report",
    "print_reverse_report",
    "print_historical_battery_report",
]
