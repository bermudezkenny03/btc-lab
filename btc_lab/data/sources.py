"""Orígenes de velas. Cada clase implementa ``CandleSource`` y se puede sustituir por otra."""

import glob
import os

import numpy as np
import pandas as pd

from btc_lab.data.cleaning import fix_duplicate_timestamps

KAGGLE_ID = "novandraanugrah/bitcoin-historical-datasets-2018-2024"


def read_candles(path: str) -> pd.DataFrame:
    """Lee un CSV de velas y deja la columna ``date`` lista para el resto del pipeline."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    date_col = next((c for c in ("open_time", "timestamp", "date", "datetime") if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"No encontré columna de fecha. Columnas: {list(df.columns)}")
    if pd.api.types.is_numeric_dtype(df[date_col]):
        unit = "ms" if df[date_col].iloc[0] > 1e11 else "s"
        df["date"] = pd.to_datetime(df[date_col], unit=unit)
    else:
        df["date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return fix_duplicate_timestamps(df)


class CsvCandleSource:
    """CSV local. Si no hay ruta, busca ``*1d*.csv`` o descarga el dataset de Kaggle."""

    def __init__(self, path: str | None = None, kaggle_id: str = KAGGLE_ID):
        self.path = path
        self.kaggle_id = kaggle_id

    def load(self) -> pd.DataFrame:
        return read_candles(self._resolve())

    def _resolve(self) -> str:
        if self.path:
            return self.path
        local = sorted(glob.glob("*1d*.csv") + glob.glob("data/*1d*.csv"))
        if local:
            return local[0]
        import kagglehub
        folder = kagglehub.dataset_download(self.kaggle_id)
        cands = sorted(glob.glob(os.path.join(folder, "*1d*.csv")))
        if not cands:
            raise FileNotFoundError(f"No encontré un CSV diario (*1d*.csv) en {folder}")
        return cands[0]


class SyntheticCandleSource:
    """Serie sintética con volatilidad agrupada, solo para probar el pipeline."""

    def __init__(self, n: int = 2500, seed: int = 42):
        self.n = n
        self.seed = seed

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n
        vol = np.zeros(n)
        vol[0] = 0.03
        ret = np.zeros(n)
        for t in range(1, n):
            vol[t] = np.sqrt(1e-5 + 0.1 * ret[t - 1] ** 2 + 0.85 * vol[t - 1] ** 2)
            ret[t] = vol[t] * rng.standard_normal()
        close = 5000 * np.exp(np.cumsum(ret))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol)))
        volume = 1e4 * np.exp(rng.normal(0, 0.3, n) + 8 * vol)
        return pd.DataFrame({
            "date": pd.date_range("2018-01-01", periods=n, freq="D"),
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
            "number_of_trades": (volume * 50).astype(int),
            "taker_buy_base_asset_volume": volume * rng.uniform(0.4, 0.6, n),
        })
