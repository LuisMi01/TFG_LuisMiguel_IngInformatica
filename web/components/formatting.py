# -*- coding: utf-8 -*-
"""Display/formatting helpers (presentation only — no calculations).

UI labels in Spanish (coherente con la memoria del TFG); code in English.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def pct(value: Optional[float], decimals: int = 2) -> str:
    """Formatea una proporción como porcentaje (0.1234 → '12.34%')."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}%}"


def num(value: Optional[float], decimals: int = 4) -> str:
    """Formatea un número con N decimales."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}"


def eur(value: Optional[float], decimals: int = 0) -> str:
    """Formatea un importe en euros con separador de miles."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f} €"


def metrics_to_table(metrics: dict, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Convierte un dict de métricas en una tabla legible de dos columnas.

    ``rows`` es una lista de ``(etiqueta, clave_en_dict, formato)`` donde
    ``formato`` ∈ {``"pct"``, ``"num"``, ``"eur"``, ``"raw"``}.
    """
    formatters = {"pct": pct, "num": num, "eur": eur, "raw": lambda v: v}
    data = []
    for label, key, fmt in rows:
        value = metrics.get(key)
        data.append({"Métrica": label, "Valor": formatters[fmt](value)})
    return pd.DataFrame(data)


def show_matplotlib(plot_fn, *args, **kwargs) -> None:
    """Renderiza una figura de ``riskpkg.viz`` en Streamlit.

    Las funciones de ``RiskVisualizer`` devuelven ``None`` y llaman a
    ``plt.show()`` (ver API_MAP.md §6/D2). Aquí las invocamos, capturamos la
    figura activa y la cerramos para no acumular memoria.
    """
    plot_fn(*args, **kwargs)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close("all")
