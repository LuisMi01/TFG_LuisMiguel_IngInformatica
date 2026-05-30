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
