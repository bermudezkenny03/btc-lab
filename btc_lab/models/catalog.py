"""Registro de modelos (abierto a extensión, cerrado a modificación de las tareas).

Cualquier estimador con ``fit`` / ``predict`` (y ``predict_proba`` si clasifica)
puede ocupar un lugar del catálogo. Las tareas recorren el catálogo; no construyen
Ridge ni Random Forest. El gráfico de importancias usa el nombre que indique
quien arma la corrida (por defecto, Random Forest).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    baseline: bool = False

    def build(self) -> Any:
        return self.factory()


class ModelCatalog:
    """Lista ordenada de modelos. ``register`` devuelve el mismo catálogo para encadenar."""

    def __init__(self, specs: list[ModelSpec] | None = None):
        self._specs: list[ModelSpec] = list(specs or [])

    def register(self, name: str, factory: Callable[[], Any], *, baseline: bool = False) -> "ModelCatalog":
        self._specs.append(ModelSpec(name, factory, baseline))
        return self

    def build(self) -> list[ModelSpec]:
        """Copia de las especificaciones. Cada ``spec.build()`` crea un estimador nuevo."""
        return list(self._specs)

    @property
    def baseline_names(self) -> set[str]:
        return {spec.name for spec in self._specs if spec.baseline}


def default_classifiers(n_rf: int, n_gb: int, seed: int) -> ModelCatalog:
    return (
        ModelCatalog()
        .register(
            "Baseline (mayoría)",
            lambda: DummyClassifier(strategy="most_frequent"),
            baseline=True,
        )
        .register(
            "Regresión logística",
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)),
        )
        .register(
            "Random Forest",
            lambda n_rf=n_rf, seed=seed: RandomForestClassifier(
                n_estimators=n_rf, min_samples_leaf=10, max_depth=6, n_jobs=-1, random_state=seed,
            ),
        )
        .register(
            "Gradient Boosting",
            lambda n_gb=n_gb, seed=seed: GradientBoostingClassifier(
                n_estimators=n_gb, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed,
            ),
        )
    )


def default_regressors(n_rf: int, n_gb: int, seed: int) -> ModelCatalog:
    return (
        ModelCatalog()
        .register(
            "Baseline (media train)",
            lambda: DummyRegressor(strategy="mean"),
            baseline=True,
        )
        .register("Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)))
        .register(
            "Random Forest",
            lambda n_rf=n_rf, seed=seed: RandomForestRegressor(
                n_estimators=n_rf, min_samples_leaf=10, max_depth=6, n_jobs=-1, random_state=seed,
            ),
        )
        .register(
            "Gradient Boosting",
            lambda n_gb=n_gb, seed=seed: GradientBoostingRegressor(
                n_estimators=n_gb, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed,
            ),
        )
    )
