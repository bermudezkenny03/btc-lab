"""Carga el histórico de Bitcoin y arma el corte temporal del laboratorio."""

from config import CSV_DEFECTO, SEMILLA  # noqa: F401  (SEMILLA vive en config, como en la guía)
from btc_lab.data.cleaning import drop_incomplete_last_candle
from btc_lab.data.sources import read_candles
from btc_lab.data.split import temporal_split
from btc_lab.features.builder import TechnicalFeatureBuilder


def cargar_velas(path=None):
    ruta = path or CSV_DEFECTO
    return drop_incomplete_last_candle(read_candles(str(ruta)))


def cargar_datos(path=None, test_size=0.2, max_train=60000):
    """Devuelve velas, matriz completa y el split train/test que ya usa lab_btc.py."""
    velas = cargar_velas(path)
    features = TechnicalFeatureBuilder()
    X_all, T_all = features.transform(velas)
    split = temporal_split(velas, X_all, T_all, test_size, max_train)
    return velas, split
