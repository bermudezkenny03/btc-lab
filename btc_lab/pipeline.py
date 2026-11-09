"""Orquesta una corrida. Arma las piezas y no implementa modelos ni indicadores."""

import os
import warnings

from btc_lab.config import ExperimentConfig
from btc_lab.data.cleaning import drop_incomplete_last_candle
from btc_lab.data.sources import CsvCandleSource, SyntheticCandleSource
from btc_lab.data.split import temporal_split
from btc_lab.features.builder import TechnicalFeatureBuilder
from btc_lab.models.catalog import default_classifiers, default_regressors
from btc_lab.reporting.plots import use_lab_style
from btc_lab.tasks.classification import ClassificationTask
from btc_lab.tasks.clustering import ClusteringTask
from btc_lab.tasks.forecast import ForecastTask
from btc_lab.tasks.regression import RegressionTask


class Experiment:
    """Composición de una corrida. Las dependencias entran por el constructor."""

    def __init__(self, config: ExperimentConfig, source, features,
                 classifiers, regressors, clustering: ClusteringTask | None = None):
        self.config = config
        self.source = source
        self.features = features
        self.classification = ClassificationTask(classifiers)
        self.regression = RegressionTask(regressors)
        self.clustering = clustering or ClusteringTask(seed=config.seed)
        self.forecast = ForecastTask(regressors)

    def run(self) -> None:
        warnings.filterwarnings("ignore")
        config = self.config
        os.makedirs(config.out, exist_ok=True)
        use_lab_style()

        candles = drop_incomplete_last_candle(self.source.load())
        print(f"Datos: {len(candles)} filas | {candles.date.min().date()} → {candles.date.max().date()}")

        X_all, T_all = self.features.transform(candles)
        split = temporal_split(candles, X_all, T_all, config.test_size, config.max_train)
        if split.n_train_before_cap > config.max_train:
            print(
                f"Aviso: se usan las {config.max_train} filas de entrenamiento más recientes "
                f"(de {split.n_train_before_cap})."
            )
        print(
            f"Train: {len(split.Xtr)} ({split.Xtr.index.min().date()}→{split.Xtr.index.max().date()}) | "
            f"Test: {len(split.Xte)} ({split.Xte.index.min().date()}→{split.Xte.index.max().date()})"
        )
        print(f"% de días que suben (train/test): {split.Ttr.y_up.mean():.1%} / {split.Tte.y_up.mean():.1%}")

        print("\n=== 1) CLASIFICACIÓN: ¿sube o baja mañana? ===")
        cls_res, cls_models, cls_best = self.classification.run(
            split.Xtr, split.Xte, split.Ttr.y_up.astype(int), split.Tte.y_up.astype(int), config.out,
        )
        print(cls_res.to_string(index=False))

        print("\n=== 2) REGRESIÓN: ¿qué tan grande es el movimiento? ===")
        reg_out, reg_best = self.regression.run(split.Xtr, split.Xte, split.Ttr, split.Tte, config.out)
        for _target, (res, _fitted, label) in reg_out.items():
            print(f"\n-- {label} --")
            print(res.to_string(index=False))

        print("\n=== 3) AGRUPACIÓN: regímenes de mercado (K-Means) ===")
        prof, k_best, sil = self.clustering.run(
            split.X, split.T, split.dates, split.close, config.out, config.k,
        )
        print(f"k elegido = {k_best} (silhouette = {sil:.3f})")
        print(prof.to_string())

        forecast = self.forecast.run(
            candles, split.X, split.T, split.X_all, cls_models, cls_best, reg_best, config.out,
        )
        if forecast is not None:
            print(
                f"\nPronóstico para la siguiente vela tras {forecast.fecha_base}: "
                f"P(sube)={forecast.prob_sube:.1%} | "
                f"movimiento esperado ≈ ±{forecast.magnitud_pct:.2f}%"
            )
        print(f"\nArchivos guardados en {config.out}/")


def run_experiment(config: ExperimentConfig) -> None:
    """Arma la corrida por defecto (CSV o serie sintética, features técnicas, catálogos base)."""
    source = SyntheticCandleSource(seed=config.seed) if config.synthetic else CsvCandleSource(config.data_path)
    experiment = Experiment(
        config,
        source,
        TechnicalFeatureBuilder(),
        default_classifiers(config.n_estimators_rf, config.n_estimators_gb, config.seed),
        default_regressors(config.n_estimators_rf, config.n_estimators_gb, config.seed),
    )
    experiment.run()
