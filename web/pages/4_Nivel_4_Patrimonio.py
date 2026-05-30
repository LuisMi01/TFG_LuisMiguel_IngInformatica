# -*- coding: utf-8 -*-
"""Nivel 4 — Patrimonio global (financiero + no financiero).

Wraps ``Level4_PatrimonyAnalyzer`` añadiendo el **rescalado D4** en capa de
presentación: ``riskpkg`` hardcodea el valor de la cartera financiera en
1 M€, así que aquí pedimos al usuario el valor real y aplicamos la misma
fórmula del analizador con ``fin_value_ref = real_value``. No se modifica
``riskpkg``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from riskpkg import RiskVisualizer  # noqa: E402
from riskpkg.levels import Level4_PatrimonyAnalyzer  # noqa: E402

from components.cache import RISKPKG_FIN_VALUE_REF, analyze_level4  # noqa: E402
from components.formatting import num, pct, show_matplotlib  # noqa: E402
from components.sidebar import render_config_summary, render_sidebar  # noqa: E402
from components.state import (  # noqa: E402
    ASSET_CLASSES,
    LIQUIDITY_LEVELS,
    NonFinancialAsset,
    default_non_financial_assets,
    get_config,
    has_config,
    set_config,
)

#: Clave de session_state donde guardamos la tabla editable de activos no
#: financieros. La sincronizamos con ``cfg.non_financial_assets`` al guardar.
NF_EDITOR_KEY = "nf_editor_df"

st.set_page_config(page_title="Nivel 4 — Patrimonio", page_icon="🏦", layout="wide")
plt.close("all")
render_sidebar()

st.title("🏦 Nivel 4 — Patrimonio global")
render_config_summary()

if not has_config():
    st.info("Configura una cartera en la barra lateral para empezar.", icon="👈")
    st.stop()

cfg = get_config()

if len(cfg.tickers) < 2:
    st.warning("La consolidación patrimonial necesita al menos 2 activos.", icon="⚠️")
    st.stop()

# ── Nota visible explicando el ajuste D4 ────────────────────────────────────
st.info(
    "**Nota sobre el rescalado patrimonial (D4).** "
    f"`riskpkg.Level4_PatrimonyAnalyzer` fija internamente el valor de la "
    f"cartera financiera en **{RISKPKG_FIN_VALUE_REF:,.0f} €** "
    "(es una simplificación MVP documentada). "
    "Para que la consolidación frente a tus activos no financieros sea coherente, "
    "esta página aplica el **rescalado `real_value / 1.000.000`** en capa de "
    "presentación reproduciendo las mismas fórmulas de `riskpkg` "
    "(pesos, contribuciones al riesgo, VaR patrimonial y descomposiciones). "
    "Las **métricas financieras** (`full_report` de la cartera) NO dependen del "
    "notional y se muestran tal cual las devuelve `riskpkg`.",
    icon="ℹ️",
)

# ══════════════════════════════════════════════════════════════════════════
# Bloque 1 — Valor monetario real de la cartera financiera
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Valor monetario actual de la cartera financiera")

c_v1, c_v2 = st.columns([2, 1])
real_value = c_v1.number_input(
    "Valor real (€)",
    min_value=0.0,
    value=float(RISKPKG_FIN_VALUE_REF),
    step=50_000.0,
    format="%.0f",
    help=(
        f"Valor monetario real de la cartera (no de mercado descargado, sino el "
        f"importe efectivamente invertido en ella). El default reproduce la "
        f"referencia que riskpkg fija internamente ({RISKPKG_FIN_VALUE_REF:,.0f} €) — "
        "en ese caso la salida coincide bit a bit con el notebook 01."
    ),
)
scale = real_value / RISKPKG_FIN_VALUE_REF if RISKPKG_FIN_VALUE_REF > 0 else 1.0
c_v2.metric(
    "Factor de rescalado",
    f"{scale:.4f}×",
    help=f"real_value / {RISKPKG_FIN_VALUE_REF:,.0f}",
)

# ══════════════════════════════════════════════════════════════════════════
# Bloque 2 — Formulario de activos no financieros (st.data_editor)
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Activos no financieros")

# Si el state aún no tiene ninguno, cargamos el preset del notebook 01.
if not cfg.non_financial_assets and NF_EDITOR_KEY not in st.session_state:
    cfg.non_financial_assets = default_non_financial_assets()
    set_config(cfg)

# DataFrame editable. Las columnas con SelectboxColumn limitan a valores válidos.
def _assets_to_df(assets: list[NonFinancialAsset]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Nombre": a.name,
                "Clase": a.asset_class,
                "Liquidez": a.liquidity,
                "Valor (€)": float(a.value),
                "Volatilidad est.": float(a.est_volatility),
                "Retorno est.": float(a.annual_return_est),
                "Correlación mercado": float(a.corr_market),
            }
            for a in assets
        ]
    )


if NF_EDITOR_KEY not in st.session_state:
    st.session_state[NF_EDITOR_KEY] = _assets_to_df(cfg.non_financial_assets)

editor_col, btn_col = st.columns([5, 1])
with btn_col:
    if st.button("Recargar preset", help="Vuelve a los 4 activos del notebook 01."):
        st.session_state[NF_EDITOR_KEY] = _assets_to_df(default_non_financial_assets())
        st.rerun()
    if st.button("Vaciar", help="Elimina todos los activos no financieros."):
        st.session_state[NF_EDITOR_KEY] = _assets_to_df([])
        st.rerun()

edited = st.data_editor(
    st.session_state[NF_EDITOR_KEY],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Nombre": st.column_config.TextColumn(required=True),
        "Clase": st.column_config.SelectboxColumn(options=ASSET_CLASSES, required=True),
        "Liquidez": st.column_config.SelectboxColumn(options=LIQUIDITY_LEVELS, required=True),
        "Valor (€)": st.column_config.NumberColumn(min_value=0.0, step=10_000.0, format="%.0f"),
        "Volatilidad est.": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
        "Retorno est.": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.005, format="%.3f"),
        "Correlación mercado": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05, format="%.2f"),
    },
    key="nf_editor",
)

# Persistir cambios en cfg.non_financial_assets antes de ejecutar el análisis.
edited = edited.dropna(subset=["Nombre", "Clase", "Liquidez"], how="any")
new_assets: list[NonFinancialAsset] = []
for _, row in edited.iterrows():
    name = (row["Nombre"] or "").strip()
    if not name:
        continue
    new_assets.append(
        NonFinancialAsset(
            name=name,
            asset_class=str(row["Clase"]),
            liquidity=str(row["Liquidez"]),
            value=float(row["Valor (€)"] or 0.0),
            est_volatility=float(row["Volatilidad est."] or 0.0),
            annual_return_est=float(row["Retorno est."] or 0.0),
            corr_market=float(row["Correlación mercado"] or 0.0),
        )
    )

if new_assets != cfg.non_financial_assets:
    cfg.non_financial_assets = new_assets
    set_config(cfg)

# ══════════════════════════════════════════════════════════════════════════
# Bloque 3 — Ejecutar análisis (Level3 → Level4 → rescalado D4)
# ══════════════════════════════════════════════════════════════════════════
# Convertir la lista a una estructura hasheable para la caché.
nf_dicts = tuple(
    tuple(item.to_riskpkg_dict().items()) for item in cfg.non_financial_assets
)

with st.spinner("Consolidando patrimonio…"):
    result = analyze_level4(
        tickers=tuple(cfg.tickers),
        weights=tuple(cfg.weights),
        start=cfg.start_iso,
        end=cfg.end_iso,
        source=cfg.data_source,
        benchmark=cfg.benchmark,
        portfolio_name=cfg.portfolio_name,
        mc_sims=cfg.mc_sims,
        mc_days=cfg.mc_days,
        return_type=cfg.return_type,
        non_financial_assets=nf_dicts,
        patrimony_name=f"Patrimonio Global · {cfg.portfolio_name}",
        real_value=real_value,
    )

# ══════════════════════════════════════════════════════════════════════════
# Bloque 4 — Visión consolidada
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Visión consolidada del patrimonio")

t1, t2, t3, t4 = st.columns(4)
t1.metric("Valor patrimonial total", f"{result.total_value:,.0f} €")
t2.metric(
    "Cartera financiera",
    f"{result.real_value:,.0f} €",
    f"{result.fin_weight:.1%} del total",
)
t3.metric(
    "Activos no financieros",
    f"{result.nf_total_value:,.0f} €",
    f"{1 - result.fin_weight:.1%} del total",
)
t4.metric(
    "VaR patrimonial est. (95%)",
    f"{result.var_patrimonial:.4f}",
    f"≈ {result.var_patrimonial * result.total_value:,.0f} €",
)

# Componente financiera (intacto, viene de riskpkg full_report)
with st.expander("Métricas de la componente financiera (sin rescalar)"):
    fm = result.financial_metrics
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Retorno anual", pct(fm["annual_return"]))
    fc2.metric("Volatilidad anual", pct(fm["volatility_ann"]))
    fc3.metric("VaR Hist. 95%", num(fm["var_hist_95"]))
    fc4.metric("Max Drawdown", pct(fm["max_drawdown"]))

# ══════════════════════════════════════════════════════════════════════════
# Bloque 5 — Detalle de activos no financieros (post-rescalado)
# ══════════════════════════════════════════════════════════════════════════
if result.nf_items:
    st.subheader("Activos no financieros (parámetros estimados)")
    df_nf = pd.DataFrame(
        [
            {
                "Nombre": it["name"],
                "Clase": Level4_PatrimonyAnalyzer.ASSET_CLASS_LABELS.get(it["class"], it["class"]),
                "Liquidez": Level4_PatrimonyAnalyzer.LIQUIDITY_LABELS.get(it["liquidity"], it["liquidity"]),
                "Valor": it["value"],
                "Peso": it["weight"],
                "Vol. est.": it["vol_est"],
                "Retorno est.": it["return_est"],
                "Corr. mercado": it["corr_mkt"],
                "Contribución riesgo": it["risk_contrib"],
            }
            for it in result.nf_items
        ]
    )
    st.dataframe(
        df_nf.style.format(
            {
                "Valor": "{:,.0f} €",
                "Peso": "{:.2%}",
                "Vol. est.": "{:.2%}",
                "Retorno est.": "{:.2%}",
                "Corr. mercado": "{:.2f}",
                "Contribución riesgo": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# Bloque 6 — Descomposiciones
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Descomposición patrimonial")

dc1, dc2 = st.columns(2)
with dc1:
    st.markdown("**Por clase de activo**")
    by_cls = pd.DataFrame(
        [
            {
                "Clase": Level4_PatrimonyAnalyzer.ASSET_CLASS_LABELS.get(c, c),
                "Valor (€)": v,
                "Peso": v / result.total_value if result.total_value > 0 else 0.0,
            }
            for c, v in sorted(result.by_class.items(), key=lambda kv: -kv[1])
        ]
    )
    st.dataframe(
        by_cls.style.format({"Valor (€)": "{:,.0f}", "Peso": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

with dc2:
    st.markdown("**Por liquidez**")
    by_liq = pd.DataFrame(
        [
            {
                "Liquidez": Level4_PatrimonyAnalyzer.LIQUIDITY_LABELS.get(l, l),
                "Valor (€)": v,
                "Peso": v / result.total_value if result.total_value > 0 else 0.0,
            }
            for l, v in sorted(result.by_liquidity.items(), key=lambda kv: -kv[1])
        ]
    )
    st.dataframe(
        by_liq.style.format({"Valor (€)": "{:,.0f}", "Peso": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# Bloque 7 — Figura
# ══════════════════════════════════════════════════════════════════════════
# Construimos el dict en el formato exacto que espera plot_patrimony_breakdown
# (mismas claves que ``Level4_PatrimonyAnalyzer._results``), usando ya los
# valores rescalados.
breakdown_dict = {
    "patrimony_name": result.patrimony_name,
    "financial_metrics": result.financial_metrics,
    "fin_value_ref": result.real_value,
    "nf_total_value": result.nf_total_value,
    "total_value": result.total_value,
    "fin_weight": result.fin_weight,
    "nf_items": result.nf_items,
    "var_patrimonial": result.var_patrimonial,
    "by_class": result.by_class,
    "by_liquidity": result.by_liquidity,
}
st.subheader("Visualización del patrimonio")
show_matplotlib(RiskVisualizer.plot_patrimony_breakdown, breakdown_dict)
