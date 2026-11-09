"""Contratos del laboratorio.

Las tareas dependen de estos protocolos (inversión de dependencias): conocen
qué necesitan, no si las velas vienen de un CSV, de Kaggle o de una serie
sintética, ni qué estimador concreto hay detrás de un nombre.
"""

from typing import Protocol

import pandas as pd


class CandleSource(Protocol):
    """Cualquier origen que entregue velas OHLCV con columna ``date``."""

    def load(self) -> pd.DataFrame:
        """DataFrame ordenado por fecha, con open, high, low, close y volume."""


class FeatureBuilder(Protocol):
    """Convierte velas en variables y targets. Sustituible por otro builder."""

    def transform(self, candles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Devuelve ``(features, targets)``. Las features solo usan el pasado."""
