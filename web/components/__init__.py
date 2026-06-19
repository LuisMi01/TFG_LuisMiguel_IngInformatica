# -*- coding: utf-8 -*-
"""Reusable web-layer components (state, sidebar, cache, formatting).

Pages stay thin: all reusable logic (config model, validation, caching,
formatting) lives here. No financial calculations — those always come from
riskpkg (see docs/WEB_SPEC.md rule #1).
"""
# Backend no interactivo: las figuras de riskpkg.viz se renderizan vía
# st.pyplot() (ver API_MAP.md D2). Debe fijarse antes de importar pyplot.
import matplotlib

matplotlib.use("Agg")

# Silencia el ruido cosmético de pandas/numpy cuando un escenario de stress
# evalúa log(precio) sobre series con NaN truncadas (ANA 2018-07 → 2022-03):
# no afecta a los resultados — riskpkg ignora esos puntos — pero contamina
# stderr. Filtro estrictamente acotado al mensaje conocido.
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in log",
)
