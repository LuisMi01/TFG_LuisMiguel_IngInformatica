# -*- coding: utf-8 -*-
"""
Detección de anomalías en series de retornos financieros.
Algoritmo: Isolation Forest (Liu, Ting & Zhou, 2008).
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..utils.constants import RANDOM_SEED


def isolation_forest_anomalies(
    returns: pd.Series,
    contamination: float = 0.02,
    random_state: int = RANDOM_SEED,
) -> pd.Series:
    """
    Detección de anomalías en series de retornos financieros.
    Identifica sesiones de estrés de mercado (fat-tail events).

    contamination: proporción esperada de anomalías (ajustar según período).
    Retorna Series con -1 (anomalía) o 1 (normal).

    Referencia: Chen et al. (2021).
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    X = returns.values.reshape(-1, 1)
    preds = model.fit_predict(X)
    return pd.Series(preds, index=returns.index, name="anomaly")
