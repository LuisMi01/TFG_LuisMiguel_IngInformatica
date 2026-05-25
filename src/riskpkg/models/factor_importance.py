# -*- coding: utf-8 -*-
"""
Análisis de importancia de factores de riesgo mediante Random Forest,
con validación cruzada walk-forward y comparación contra la contribución
marginal al riesgo clásica (correlación de Spearman).
"""
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from ..utils.constants import RANDOM_SEED


def rf_risk_factor_importance(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Dict:
    """
    Estima la contribución relativa de cada activo al riesgo de la cartera
    mediante feature importance de Random Forest.

    Usa Time Series Cross-Validation (walk-forward) para evitar data leakage.
    Compara el ranking con la contribución marginal al riesgo clásica.

    La correlación de Spearman entre ambos rankings > 0.70 indica coherencia
    entre el enfoque ML y el método clásico.

    Retorna dict con importancias, ranking RF, ranking clásico y Spearman.
    """
    # Alinear fechas
    common_idx = asset_returns.index.intersection(portfolio_returns.index)
    X = asset_returns.loc[common_idx]
    y = portfolio_returns.loc[common_idx].abs()   # magnitud del retorno

    # Walk-forward cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances_list = []

    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        importances_list.append(rf.feature_importances_)

    # Promedio de importancias sobre todos los folds
    mean_importances = np.mean(importances_list, axis=0)
    rf_series = pd.Series(
        mean_importances,
        index=X.columns,
        name="RF_importance",
    ).sort_values(ascending=False)

    return {
        "rf_importances":  rf_series,
        "rf_ranking":      list(rf_series.index),
        "n_cv_splits":     n_splits,
    }


def compare_risk_attribution(
    rf_importances: pd.Series,
    classical_mrc: pd.Series,
) -> Dict:
    """
    Compara el ranking de importancia RF con la contribución marginal clásica.
    Calcula correlación de Spearman entre ambos rankings.

    Criterio de aceptación (TFG): Spearman > 0.70.
    """
    # Alinear índices
    common = rf_importances.index.intersection(classical_mrc.index)
    rf_aligned = rf_importances.loc[common]
    mrc_aligned = classical_mrc.loc[common]

    spearman_corr, p_value = stats.spearmanr(rf_aligned.values, mrc_aligned.values)

    return {
        "spearman_correlation": round(spearman_corr, 4),
        "p_value":              round(p_value, 4),
        "n_assets":             len(common),
        "meets_criterion":      spearman_corr > 0.70,
        "interpretation": (
            "✅  Coherencia RF–Clásico validada (Spearman > 0.70)"
            if spearman_corr > 0.70 else
            "⚠️  Baja coherencia RF–Clásico (Spearman ≤ 0.70)"
        ),
    }
