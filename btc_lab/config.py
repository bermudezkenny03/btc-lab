"""Parámetros de una corrida. No entrena ni lee archivos."""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    data_path: str | None = None
    synthetic: bool = False
    test_size: float = 0.2
    k: int | None = None
    out: str = "resultados"
    max_train: int = 60000
    fast: bool = False
    seed: int = 42

    @property
    def n_estimators_rf(self) -> int:
        return 100 if self.fast else 300

    @property
    def n_estimators_gb(self) -> int:
        return 80 if self.fast else 150
