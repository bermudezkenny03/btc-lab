"""Variables técnicas calculadas solo con información pasada (sin fuga hacia el target)."""

import numpy as np
import pandas as pd

from btc_lab.features.indicators import rsi


class TechnicalFeatureBuilder:
    """Features de retorno, volatilidad, tendencia, vela y calendario, más los targets de la vela siguiente."""

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
        lr = np.log(c).diff()
        f = pd.DataFrame(index=df.index)
        f["ret_1"] = lr
        for n in (2, 5, 10, 20):
            f[f"ret_{n}"] = np.log(c).diff(n)
        for n in (5, 10, 20):
            f[f"vol_{n}"] = lr.rolling(n).std()
        for n in (10, 20, 50):
            f[f"sma_ratio_{n}"] = c / c.rolling(n).mean() - 1
        f["rsi"] = rsi(c)
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
        ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
        f["boll_pct_b"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
        f["range"] = (h - l) / c
        f["body"] = (c - o) / o
        f["upper_wick"] = (h - np.maximum(o, c)) / c
        f["lower_wick"] = (np.minimum(o, c) - l) / c
        f["vol_chg"] = np.log(v.replace(0, np.nan)).diff()
        f["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
        if "number_of_trades" in df:
            f["trades_chg"] = np.log(df["number_of_trades"].replace(0, np.nan)).diff()
        if "taker_buy_base_asset_volume" in df:
            f["taker_ratio"] = df["taker_buy_base_asset_volume"] / v
        dow = df["date"].dt.dayofweek
        f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
        if df["date"].diff().median() < pd.Timedelta(days=1):
            hr = df["date"].dt.hour + df["date"].dt.minute / 60
            f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)

        nxt = lr.shift(-1)
        t = pd.DataFrame({
            "next_ret": nxt,
            "y_up": (nxt > 0).astype(float).where(nxt.notna()),
            "next_abs": nxt.abs(),
        })
        return f.replace([np.inf, -np.inf], np.nan), t
