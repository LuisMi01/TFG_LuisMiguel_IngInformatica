"""
Test de constantes globales.

CLAUDE.md sección 9: RANDOM_SEED no debe cambiar sin avisar — su valor está
fijado para garantizar la reproducibilidad de los outputs documentados.
"""
import riskpkg
from riskpkg.utils import constants


def test_random_seed_es_42():
    assert constants.RANDOM_SEED == 42


def test_trading_days_year_252():
    assert constants.TRADING_DAYS_YEAR == 252


def test_default_confidence_95():
    assert constants.DEFAULT_CONFIDENCE == 0.95


def test_min_observations_252():
    assert constants.MIN_OBSERVATIONS == 252


def test_benchmark_ticker_spy():
    assert constants.BENCHMARK_TICKER == "SPY"


def test_constantes_reexportadas_en_paquete_raiz():
    """El __init__ del paquete reexporta las constantes — uso conveniente."""
    assert riskpkg.RANDOM_SEED == constants.RANDOM_SEED
    assert riskpkg.TRADING_DAYS_YEAR == constants.TRADING_DAYS_YEAR
    assert riskpkg.DEFAULT_CONFIDENCE == constants.DEFAULT_CONFIDENCE
    assert riskpkg.BENCHMARK_TICKER == constants.BENCHMARK_TICKER
