"""Pronóstico de la vela siguiente, reentrenando los mejores modelos con todo el histórico válido."""

import json
from dataclasses import dataclass


@dataclass
class NextCandleForecast:
    fecha_base: str
    prob_sube: float
    magnitud_pct: float
    modelo_clasificacion: str
    modelo_magnitud: str

    def as_dict(self) -> dict:
        return {
            "fecha_base": self.fecha_base,
            "prob_sube": round(float(self.prob_sube), 4),
            "magnitud_esperada_pct": round(float(self.magnitud_pct), 3),
            "modelo_clasificacion": self.modelo_clasificacion,
            "modelo_magnitud": self.modelo_magnitud,
        }


class ForecastTask:
    def __init__(self, regressors):
        self.regressors = regressors

    def run(self, candles, X, T, X_all, cls_models, cls_best: str, reg_best: str, outdir: str):
        xl = X_all.iloc[[-1]][X.columns]
        if not xl.notna().all(axis=1).iloc[0]:
            return None
        classifier = cls_models[cls_best]
        classifier.fit(X, T.y_up.astype(int))
        regressor = next(spec.build() for spec in self.regressors.build() if spec.name == reg_best)
        regressor.fit(X, T.next_abs)
        forecast = NextCandleForecast(
            fecha_base=str(candles.date.iloc[-1]),
            prob_sube=float(classifier.predict_proba(xl)[0, 1]),
            magnitud_pct=float(regressor.predict(xl)[0] * 100),
            modelo_clasificacion=cls_best,
            modelo_magnitud=reg_best,
        )
        with open(f"{outdir}/pronostico_manana.json", "w", encoding="utf-8") as fh:
            json.dump(forecast.as_dict(), fh, indent=2, ensure_ascii=False)
        return forecast
