"""Regresión: magnitud y retorno con signo de la siguiente vela."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from btc_lab.models.catalog import ModelCatalog
from btc_lab.reporting.plots import save_magnitude


class RegressionTask:
    def __init__(self, catalog: ModelCatalog):
        self.catalog = catalog

    def run(self, Xtr, Xte, t_tr, t_te, outdir: str):
        out = {}
        targets = (("next_ret", "Retorno con signo"), ("next_abs", "Magnitud |retorno|"))
        for target, label in targets:
            rows, fitted = [], {}
            for spec in self.catalog.build():
                model = spec.build()
                model.fit(Xtr, t_tr[target])
                pred = model.predict(Xte)
                row = {
                    "modelo": spec.name,
                    "MAE_%": mean_absolute_error(t_te[target], pred) * 100,
                    "RMSE_%": np.sqrt(mean_squared_error(t_te[target], pred)) * 100,
                    "R2": r2_score(t_te[target], pred),
                }
                if target == "next_ret":
                    row["acierto_direccion"] = (np.sign(pred) == np.sign(t_te[target])).mean()
                rows.append(row)
                fitted[spec.name] = (model, pred)
            res = pd.DataFrame(rows).round(4)
            res.to_csv(f"{outdir}/regresion_{target}_metricas.csv", index=False)
            out[target] = (res, fitted, label)

        res, fitted, _ = out["next_abs"]
        ranked = res[~res.modelo.isin(self.catalog.baseline_names)].sort_values("R2", ascending=False)
        best = ranked.iloc[0].modelo
        pred = fitted[best][1]
        save_magnitude(
            t_te.index, t_te["next_abs"], pred, best, f"{outdir}/regresion_magnitud.png",
        )
        return out, best
