"""Tests del catálogo de escenarios históricos de estrés."""
from dataclasses import FrozenInstanceError

import pytest

from riskpkg.stress.scenarios import (
    HISTORICAL_SCENARIOS,
    StressWindow,
    get_scenario,
)


def test_catalogo_contiene_los_ocho_escenarios_documentados():
    """CLAUDE.md §3 declara 8 escenarios: dot-com, GFC, Eurozona, China,
    Volmageddon, COVID, 2022 (inflación) y SVB."""
    esperados = {
        "dot_com_2000", "gfc_2008", "eurozone_2011", "china_2015",
        "volmageddon_2018", "covid_2020", "inflation_2022", "svb_2023",
    }
    assert esperados.issubset(set(HISTORICAL_SCENARIOS.keys()))


def test_stress_window_es_inmutable():
    """@dataclass(frozen=True) — no debe permitir mutar campos."""
    s = HISTORICAL_SCENARIOS["gfc_2008"]
    with pytest.raises(FrozenInstanceError):
        s.start = "2099-01-01"   # type: ignore[misc]


def test_get_scenario_devuelve_stress_window():
    s = get_scenario("covid_2020")
    assert isinstance(s, StressWindow)
    assert s.key == "covid_2020"
    assert s.start == "2020-02-19"
    assert s.end == "2020-03-23"


def test_get_scenario_lanza_keyerror_con_clave_invalida():
    with pytest.raises(KeyError, match="no encontrado"):
        get_scenario("does_not_exist")


def test_todas_las_ventanas_tienen_fechas_iso_y_descripcion():
    """Validación estructural del catálogo: campos obligatorios no vacíos."""
    for key, s in HISTORICAL_SCENARIOS.items():
        assert s.key == key
        assert len(s.start) == 10 and s.start[4] == "-"
        assert len(s.end) == 10 and s.end[4] == "-"
        assert s.start < s.end
        assert s.name.strip() != ""
        assert s.description.strip() != ""
