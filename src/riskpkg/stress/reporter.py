# -*- coding: utf-8 -*-
"""
Utilidades de presentación de resultados de stress testing.
Tres funciones print_* simétricas a las usadas en los analizadores de nivel.
"""
from typing import Dict

import pandas as pd


def print_historical_report(result: Dict) -> None:
    """Imprime el informe de un stress test histórico."""
    print(f"\n{'═'*68}")
    print(f"  STRESS TEST HISTÓRICO — {result['scenario_name']}")
    print(f"{'═'*68}")
    print(f"  Período           : {result['start']} → {result['end']} "
          f"({result['n_sessions']} sesiones)")
    print(f"  Contexto          : {result['scenario_description']}")
    print(f"{'─'*68}")
    print(f"  Valor inicial     : {result['initial_value']:>12,.0f} €")
    print(f"  Valor final       : {result['portfolio_value_final']:>12,.0f} €")
    print(f"  Valor mínimo      : {result['portfolio_value_min']:>12,.0f} €")
    print(f"  P&L cartera       : {result['portfolio_pnl_pct']:>12.2%}")
    print(f"{'─'*68}")
    print(f"  Max Drawdown      : {result['max_drawdown_pct']:>12.2%}")
    print(f"  Días hasta suelo  : {result['days_to_trough']:>12} sesiones")
    rec = "Sí" if result["recovered_within_window"] else "No (ventana cerrada en pérdida)"
    print(f"  Recuperado        : {rec}")
    if result["tickers_excluded"]:
        print(f"{'─'*68}")
        print(f"  Activos excluidos (sin datos en ventana): {result['tickers_excluded']}")
    print(f"{'─'*68}")
    print(f"  P&L individual por activo (pesos renormalizados):")
    for t, pnl in result["asset_pnl_pct"].items():
        w = result["weights_effective"].get(t, 0)
        bar_len = max(0, int(-pnl * 50))
        bar = "▓" * min(bar_len, 30)
        print(f"    {t:>8}  peso: {w:>5.1%}  P&L: {pnl:>+7.2%}  {bar}")
    print(f"{'═'*68}\n")


def print_hypothetical_report(result: Dict) -> None:
    """Imprime el informe de un stress test hipotético."""
    print(f"\n{'═'*68}")
    print(f"  STRESS TEST HIPOTÉTICO — {result['shock_name']}")
    print(f"{'═'*68}")
    print(f"  Descripción       : {result['shock_description']}")
    print(f"{'─'*68}")
    print(f"  Valor inicial     : {result['initial_value']:>12,.0f} €")
    print(f"  Valor tras shock  : {result['portfolio_value_after']:>12,.0f} €")
    print(f"  P&L cartera (%)   : {result['portfolio_pnl_pct']:>12.2%}")
    print(f"  P&L cartera (€)   : {result['portfolio_pnl_eur']:>12,.0f} €")
    print(f"{'─'*68}")
    print(f"  Shock aplicado y contribución por activo:")
    for t in result["asset_shocks"]:
        shock = result["asset_shocks"][t]
        contrib = result["asset_contributions"][t]
        cls = result["ticker_to_class"].get(t, "—")
        print(f"    {t:>8}  [{cls:<12}] shock: {shock:>+7.2%}  → contrib: {contrib:>+9,.0f} €")
    print(f"{'═'*68}\n")


def print_reverse_report(result: Dict) -> None:
    """Imprime el informe de un reverse stress test."""
    print(f"\n{'═'*68}")
    print(f"  REVERSE STRESS TEST")
    print(f"{'═'*68}")
    print(f"  Pérdida objetivo  : {result['target_loss']:>12.2%}")
    print(f"  Horizonte         : {result['horizon']}")
    print(f"  σ de cartera      : {result['sigma_portfolio']:>12.4f}")
    print(f"{'─'*68}")
    print(f"  Distancia de Mahalanobis : {result['mahalanobis_distance']:>10.4f} σ")
    print(f"  Probabilidad (cola izq.) : {result['plausibility_prob']:>10.4%}")
    print(f"  Verificación pérdida     : {-result['portfolio_loss_check']:>10.4%}  "
          f"(debe coincidir con objetivo)")

    # Interpretación cualitativa de plausibilidad
    z = result['mahalanobis_distance']
    if z < 2:
        interp = "✅  Plausible (< 2σ): escenario consistente con régimen normal"
    elif z < 3:
        interp = "⚠️  Moderado (2-3σ): escenario de cola, observable cada ~100 sesiones"
    elif z < 4:
        interp = "⚠️  Extremo (3-4σ): probabilidad ~1/15.000 bajo normalidad"
    else:
        interp = "🚨  Tail extremo (> 4σ): casi imposible bajo normalidad, requiere modelo de colas pesadas"
    print(f"  Interpretación    : {interp}")
    print(f"{'─'*68}")
    print(f"  Vector de shock óptimo por activo:")
    for t, s in result["shock_vector"].items():
        bar_len = max(0, int(-s * 100))
        bar = "▓" * min(bar_len, 30)
        print(f"    {t:>8}  {s:>+8.4f}  ({s:>+.2%})  {bar}")
    print(f"{'═'*68}\n")


def print_historical_battery_report(battery: pd.DataFrame) -> None:
    """Imprime el informe consolidado de una batería de stress tests históricos."""
    print(f"\n{'═'*78}")
    print(f"  BATERÍA DE STRESS TESTS HISTÓRICOS — Resumen comparativo")
    print(f"{'═'*78}")
    print(f"  {'Escenario':<42} {'P&L':>9} {'MaxDD':>9} {'Suelo':>7}")
    print(f"  {'-'*42} {'-'*9} {'-'*9} {'-'*7}")
    for scenario, row in battery.iterrows():
        if "error" in row and pd.notna(row.get("error", pd.NA)):
            print(f"  {scenario:<42} ERROR: {row['error']}")
            continue
        pnl = row["pnl_pct"]
        mdd = row["max_drawdown_pct"]
        dtt = int(row["days_to_trough"])
        flag = " ⚠" if pnl < -0.20 else ("  " if pnl < 0 else " ✓")
        print(f"  {scenario:<42} {pnl:>+9.2%} {mdd:>+9.2%} {dtt:>7d}{flag}")
    print(f"{'═'*78}\n")
