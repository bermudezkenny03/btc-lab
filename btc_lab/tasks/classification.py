"""Clasificación: ¿la siguiente vela sube o baja?"""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from btc_lab.models.catalog import ModelCatalog
from btc_lab.reporting.plots import save_confusion, save_importances


class ClassificationTask:
    def __init__(self, catalog: ModelCatalog, importance_model: str = "Random Forest"):
        self.catalog = catalog
        self.importance_model = importance_model

    def run(self, Xtr, Xte, ytr, yte, outdir: str):
        tscv = TimeSeriesSplit(n_splits=5)
        rows: list[dict] = []
        fitted: dict[str, Any] = {}
        for spec in self.catalog.build():
            model = spec.build()
            cv_auc = 0.5 if spec.baseline else cross_val_score(
                model, Xtr, ytr, cv=tscv, scoring="roc_auc"
            ).mean()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)[:, 1]
            rows.append({
                "modelo": spec.name,
                "accuracy": accuracy_score(yte, pred),
                "balanced_acc": balanced_accuracy_score(yte, pred),
                "f1": f1_score(yte, pred, zero_division=0),
                "roc_auc_test": roc_auc_score(yte, proba) if len(set(proba)) > 1 else 0.5,
                "roc_auc_cv_train": cv_auc,
            })
            fitted[spec.name] = model
        res = pd.DataFrame(rows).round(4)
        res.to_csv(f"{outdir}/clasificacion_metricas.csv", index=False)

        ranked = res[~res.modelo.isin(self.catalog.baseline_names)].sort_values(
            "roc_auc_test", ascending=False
        )
        best = ranked.iloc[0].modelo
        cm = confusion_matrix(yte, fitted[best].predict(Xte))
        save_confusion(cm, best, f"{outdir}/clasificacion_confusion.png")

        ranked_model = fitted.get(self.importance_model)
        if ranked_model is not None and hasattr(ranked_model, "feature_importances_"):
            imp = pd.Series(ranked_model.feature_importances_, index=Xtr.columns).sort_values().tail(12)
            save_importances(imp, f"{outdir}/clasificacion_importancias.png", self.importance_model)
        return res, fitted, best
