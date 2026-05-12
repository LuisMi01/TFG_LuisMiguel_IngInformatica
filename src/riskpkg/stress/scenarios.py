# -*- coding: utf-8 -*-
"""
Catálogo de ventanas históricas de crisis financieras.

Cada escenario está definido por su fecha de inicio (pico previo o evento
detonante), fecha de fin (mínimo de mercado o resolución del episodio) y
una breve descripción contextual. Las fechas se han seleccionado con base
en hitos documentados de los índices S&P 500 y MSCI World.

Referencias generales:
    Reinhart & Rogoff (2009). This Time Is Different.
    BIS Annual Report (varios años).
    EBA Stress Test Methodology (2023).
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class StressWindow:
    """Definición inmutable de una ventana histórica de estrés de mercado."""
    key: str               # Identificador estable (snake_case)
    name: str              # Nombre legible
    start: str             # ISO 8601 (YYYY-MM-DD)
    end: str               # ISO 8601 (YYYY-MM-DD)
    description: str       # Contexto en 1-2 frases


# ─── Catálogo curado de escenarios históricos ───────────────────────────────
HISTORICAL_SCENARIOS: Dict[str, StressWindow] = {
    "dot_com_2000": StressWindow(
        key="dot_com_2000",
        name="Burbuja .com (2000–2002)",
        start="2000-03-10",
        end="2002-10-09",
        description=(
            "Estallido de la burbuja tecnológica. El NASDAQ cae un 78% desde el "
            "pico del 10 de marzo de 2000 hasta el mínimo del 9 de octubre de 2002."
        ),
    ),
    "gfc_2008": StressWindow(
        key="gfc_2008",
        name="Crisis Financiera Global (Lehman)",
        start="2008-09-01",
        end="2009-03-09",
        description=(
            "Quiebra de Lehman Brothers (15 sept. 2008) y crisis sistémica de "
            "liquidez. El S&P 500 alcanza su mínimo el 9 de marzo de 2009 tras "
            "una caída del 57% desde octubre de 2007."
        ),
    ),
    "eurozone_2011": StressWindow(
        key="eurozone_2011",
        name="Crisis de deuda soberana europea",
        start="2011-07-01",
        end="2011-10-04",
        description=(
            "Rescate de Grecia y contagio a Italia y España. Pico de prima de "
            "riesgo y mínimo de mercado europeo el 4 de octubre de 2011."
        ),
    ),
    "china_2015": StressWindow(
        key="china_2015",
        name="Devaluación del yuan y Black Monday",
        start="2015-08-01",
        end="2016-02-11",
        description=(
            "Devaluación sorpresa del yuan (11 ago.) y desplome de bolsas globales "
            "el 24 de agosto (Black Monday). Mínimo el 11 de febrero de 2016."
        ),
    ),
    "volmageddon_2018": StressWindow(
        key="volmageddon_2018",
        name="Volmageddon (febrero 2018)",
        start="2018-01-26",
        end="2018-04-02",
        description=(
            "Explosión del VIX el 5 de febrero de 2018 (de 17 a 37 en una sesión). "
            "Colapso de ETPs vinculados a volatilidad inversa."
        ),
    ),
    "covid_2020": StressWindow(
        key="covid_2020",
        name="Crash COVID-19",
        start="2020-02-19",
        end="2020-03-23",
        description=(
            "Pandemia y confinamientos globales. El S&P 500 cae un 34% en 33 "
            "sesiones, su crash más rápido en la historia."
        ),
    ),
    "inflation_2022": StressWindow(
        key="inflation_2022",
        name="Inflación y guerra de Ucrania (2022)",
        start="2022-01-03",
        end="2022-10-12",
        description=(
            "Endurecimiento monetario por inflación (8-9% interanual), guerra de "
            "Ucrania y mercado bajista simultáneo en renta fija y variable."
        ),
    ),
    "svb_2023": StressWindow(
        key="svb_2023",
        name="Mini-crisis bancaria (SVB / Credit Suisse)",
        start="2023-03-08",
        end="2023-03-24",
        description=(
            "Caídas de Silicon Valley Bank y Signature Bank (10 mar.) y rescate "
            "de Credit Suisse por UBS (19 mar.). Pánico bancario regional en EEUU."
        ),
    ),
}


def list_scenarios() -> None:
    """Imprime el catálogo de escenarios disponibles de forma legible."""
    print(f"\n{'═'*70}")
    print("  Catálogo de Escenarios Históricos de Estrés")
    print(f"{'═'*70}")
    for s in HISTORICAL_SCENARIOS.values():
        print(f"  [{s.key:<20}] {s.name}")
        print(f"     {s.start} → {s.end}")
        print(f"     {s.description}")
        print()
    print(f"{'═'*70}\n")


def get_scenario(key: str) -> StressWindow:
    """Obtiene un escenario por clave. Lanza KeyError si no existe."""
    if key not in HISTORICAL_SCENARIOS:
        available = ", ".join(HISTORICAL_SCENARIOS.keys())
        raise KeyError(f"Escenario '{key}' no encontrado. Disponibles: {available}")
    return HISTORICAL_SCENARIOS[key]
