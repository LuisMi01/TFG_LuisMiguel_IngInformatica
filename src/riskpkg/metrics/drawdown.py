# -*- coding: utf-8 -*-
"""
Métricas de drawdown: máxima caída y período de recuperación.
"""
from typing import Optional

import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    """Máxima caída pico a valle (valor negativo)."""
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak) - 1
    return dd.min()


def drawdown_recovery_period(returns: pd.Series) -> Optional[int]:
    """
    Período de recuperación tras el máximo drawdown.
    Retorna el número de sesiones desde el mínimo hasta recuperar el pico,
    o None si aún no se ha recuperado al final de la serie.
    """
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak) - 1
    trough_idx = int(dd.values.argmin())

    # Buscar si la serie supera el pico previo tras el mínimo
    peak_before_trough = peak.iloc[trough_idx]
    recovery = cum.iloc[trough_idx:][cum.iloc[trough_idx:] >= peak_before_trough]
    if recovery.empty:
        return None    # Aún no recuperado
    return int(recovery.index.get_loc(recovery.index[0]) if hasattr(recovery.index, 'get_loc')
               else len(cum.iloc[trough_idx:recovery.index[0]]))
