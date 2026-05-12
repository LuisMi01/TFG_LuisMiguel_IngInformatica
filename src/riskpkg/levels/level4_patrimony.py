# -*- coding: utf-8 -*-
"""
NIVEL 4 — Patrimonio global: integración de cartera financiera y
activos no financieros (inmuebles, alternativos, efectivo).
"""
from typing import Dict, List, Optional

from .level3_portfolio import Level3_PortfolioAnalyzer


class Level4_PatrimonyAnalyzer:
    """
    NIVEL 4: Análisis del riesgo del patrimonio global del inversor.

    Integra la cartera financiera (Nivel 3) con activos no financieros:
    inmuebles, bonos no cotizados, activos alternativos, etc.

    Los activos no financieros se modelan con parámetros exógenos estimados
    (volatilidad, retorno esperado, correlación con mercado) ya que no cotizan.

    Incorpora:
      · Agregación patrimonial por valor de mercado estimado
      · Descomposición del riesgo por clase de activo
      · Descomposición del riesgo por nivel de liquidez
      · VaR estimado del patrimonio global integrado

    Estructura de activos no financieros (dict):
        {
            "name": "Inmueble Madrid",
            "class": "real_estate",         # "real_estate"|"bond"|"alternative"|"cash"
            "liquidity": "illiquid",        # "liquid"|"semi_liquid"|"illiquid"
            "value": 300_000,
            "annual_return_est": 0.03,
            "volatility_est": 0.06,
            "correlation_with_market": 0.30,
        }
    """

    LIQUIDITY_LABELS = {
        "liquid":      "Líquido (< 1 semana)",
        "semi_liquid": "Semilíquido (1 sem – 6 meses)",
        "illiquid":    "Ilíquido (> 6 meses)",
    }

    ASSET_CLASS_LABELS = {
        "equity":      "Renta Variable",
        "fixed_income": "Renta Fija",
        "real_estate": "Inmobiliario",
        "commodity":   "Materias Primas",
        "alternative": "Activos Alternativos",
        "cash":        "Efectivo / Equivalentes",
    }

    def __init__(
        self,
        financial_portfolio: Level3_PortfolioAnalyzer,
        non_financial_assets: Optional[List[Dict]] = None,
        patrimony_name: str = "Patrimonio Global",
    ):
        self.financial_portfolio = financial_portfolio
        self.non_financial_assets = non_financial_assets or []
        self.patrimony_name = patrimony_name
        self._results: Optional[Dict] = None

    def run(self) -> "Level4_PatrimonyAnalyzer":
        """Integra componente financiera y no financiera en visión global."""

        fp = self.financial_portfolio

        # Valor de mercado de la cartera financiera (última sesión, base 1€ por €invertido)
        # Usamos el VaR al 95% para estimar la pérdida máxima esperada de la cartera financiera
        fm = fp._port_metrics

        # Activos no financieros
        nf_total_value = sum(a["value"] for a in self.non_financial_assets)
        fin_value_ref = 1_000_000   # Valor de referencia de la cartera financiera (ajustar)

        total_value = fin_value_ref + nf_total_value

        # Peso relativo de cada componente
        fin_weight = fin_value_ref / total_value if total_value > 0 else 1.0

        # Volatilidad estimada del patrimonio global (usando correlación asumida)
        port_vol = fm["volatility_ann"]
        nf_items = []
        for a in self.non_financial_assets:
            w_nf = a["value"] / total_value
            vol_a = a.get("volatility_est", 0.0)
            corr = a.get("correlation_with_market", 0.0)
            ret_a = a.get("annual_return_est", 0.0)
            nf_items.append({
                "name":      a["name"],
                "class":     a.get("class", "alternative"),
                "liquidity": a.get("liquidity", "illiquid"),
                "value":     a["value"],
                "weight":    w_nf,
                "vol_est":   vol_a,
                "return_est": ret_a,
                "corr_mkt":  corr,
                # Contribución al riesgo: w_i * vol_i * corr_i (simplificado)
                "risk_contrib": w_nf * vol_a * corr,
            })

        # VaR estimado del patrimonio global (suma ponderada como aproximación)
        var_patrimonial = (
            fin_weight * fm["var_hist_95"] +
            sum(a["risk_contrib"] * 1.645 for a in nf_items)
        )

        # Desglose por clase de activo y por liquidez
        by_class = {}
        by_liquidity = {}
        for a in nf_items:
            cls = a["class"]
            liq = a["liquidity"]
            by_class[cls] = by_class.get(cls, 0) + a["value"]
            by_liquidity[liq] = by_liquidity.get(liq, 0) + a["value"]

        # Cartera financiera se clasifica como equity/liquid
        by_class["equity"] = by_class.get("equity", 0) + fin_value_ref
        by_liquidity["liquid"] = by_liquidity.get("liquid", 0) + fin_value_ref

        self._results = {
            "patrimony_name":   self.patrimony_name,
            "financial_metrics": fm,
            "fin_value_ref":    fin_value_ref,
            "nf_total_value":   nf_total_value,
            "total_value":      total_value,
            "fin_weight":       fin_weight,
            "nf_items":         nf_items,
            "var_patrimonial":  var_patrimonial,
            "by_class":         by_class,
            "by_liquidity":     by_liquidity,
        }
        return self

    def print_report(self) -> None:
        if self._results is None:
            raise RuntimeError("Ejecuta run() primero.")

        r = self._results
        fm = r["financial_metrics"]

        print(f"\n{'═'*65}")
        print(f"  NIVEL 4 — {r['patrimony_name']}")
        print(f"{'═'*65}")
        print(f"  Valor Patrimonial Total    : {r['total_value']:>14,.0f} €")
        print(f"    Cartera Financiera       : {r['fin_value_ref']:>14,.0f} €  "
              f"({r['fin_weight']:.1%})")
        print(f"    Activos No Financieros   : {r['nf_total_value']:>14,.0f} €  "
              f"({1-r['fin_weight']:.1%})")
        print(f"{'─'*65}")
        print(f"  ▸ Componente Financiera (Cartera)")
        print(f"    Retorno anual estimado   : {fm['annual_return']:>10.2%}")
        print(f"    Volatilidad anual        : {fm['volatility_ann']:>10.2%}")
        print(f"    VaR Histórico (95%)      : {fm['var_hist_95']:>10.4f}")
        print(f"    Sharpe Ratio             : {fm['sharpe_ratio']:>10.4f}")
        print(f"    Max Drawdown             : {fm['max_drawdown']:>10.2%}")

        if r["nf_items"]:
            print(f"{'─'*65}")
            print(f"  ▸ Activos No Financieros (parámetros estimados)")
            for a in r["nf_items"]:
                liq_label = self.LIQUIDITY_LABELS.get(a["liquidity"], a["liquidity"])
                cls_label = self.ASSET_CLASS_LABELS.get(a["class"], a["class"])
                print(f"    {a['name']:<28}  Valor: {a['value']:>10,.0f} €")
                print(f"      Clase: {cls_label:<20}  Liquidez: {liq_label}")
                print(f"      Retorno est: {a['return_est']:.1%}  "
                      f"Vol.est: {a['vol_est']:.1%}  "
                      f"Corr.mkt: {a['corr_mkt']:.2f}")

        print(f"{'─'*65}")
        print(f"  ▸ Descomposición por Clase de Activo:")
        for cls, val in sorted(r["by_class"].items(), key=lambda x: -x[1]):
            label = self.ASSET_CLASS_LABELS.get(cls, cls)
            pct = val / r["total_value"] * 100
            bar = "█" * max(1, int(pct / 4))
            print(f"    {label:<25} {val:>12,.0f} €  {pct:>5.1f}%  {bar}")

        print(f"{'─'*65}")
        print(f"  ▸ Descomposición por Nivel de Liquidez:")
        for liq, val in sorted(r["by_liquidity"].items(), key=lambda x: -x[1]):
            label = self.LIQUIDITY_LABELS.get(liq, liq)
            pct = val / r["total_value"] * 100
            bar = "█" * max(1, int(pct / 4))
            print(f"    {label:<35} {val:>12,.0f} €  {pct:>5.1f}%  {bar}")

        print(f"{'─'*65}")
        print(f"  VaR Patrimonial est. (95%)  : {r['var_patrimonial']:.4f}  "
              f"(aprox. {r['var_patrimonial']*r['total_value']:,.0f} €)")
        print(f"{'═'*65}\n")
