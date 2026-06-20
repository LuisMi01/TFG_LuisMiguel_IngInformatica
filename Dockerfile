# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Imagen reproducible para el dashboard del TFG (Streamlit + riskpkg).
# Objetivo: resultados bit a bit idénticos entre Linux (dev) y Windows (backup
# de defensa). Python 3.11-slim-bookworm fija el intérprete y el sistema base.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential es necesario para compilar las extensiones Cython de `arch`
# (modelos GARCH/GJR-GARCH) cuando no hay wheel disponible para la combinación
# concreta de Python/arch. El resto de dependencias compiladas (numpy, scipy,
# scikit-learn, statsmodels, pandas) llegan como wheels manylinux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Capa de cache de dependencias ───────────────────────────────────────────
# Copiamos solo los metadatos del paquete y un esqueleto mínimo de `src/`
# para que `pip install -e .` resuelva y descargue TODAS las dependencias
# declaradas en pyproject.toml. Esta capa se reaprovecha mientras pyproject
# no cambie, lo que acelera enormemente las reconstrucciones.
COPY pyproject.toml README.md ./
RUN mkdir -p src/riskpkg \
    && printf '__version__ = "0.5.0"\n' > src/riskpkg/__init__.py \
    && pip install -e . streamlit

# ── Código real ─────────────────────────────────────────────────────────────
# Sustituye el esqueleto por el contenido completo del repositorio. La
# instalación editable apunta a /app/src/riskpkg, así que el `riskpkg` que
# se importa ya es el código real sin reinstalar nada.
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "web/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
