"""Nombres del experimento y del almacén. Un solo lugar, como en la guía de MLflow."""

import os
import sys
from pathlib import Path

import mlflow

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

EXPERIMENTO = "prediccion-btc"
EXPERIMENTO_MAGNITUD = "magnitud-btc"
EXPERIMENTO_REGIMENES = "regimenes-btc"
MODELO_REGISTRADO = "clasificador-btc"
SEMILLA = 42
CSV_DEFECTO = RAIZ / "data" / "btc_1d_data_2018_to_2025.csv"


def usar_almacen() -> None:
    """SQLite del proyecto, salvo que MLFLOW_TRACKING_URI ya venga definida."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        uri = "sqlite:///" + (RAIZ / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
