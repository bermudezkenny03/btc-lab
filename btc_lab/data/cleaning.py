"""Limpieza de velas. No calcula variables ni entrena modelos."""

import numpy as np
import pandas as pd


def fix_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """El dataset trae velas duplicadas con precios corruptos (~108 000 USD en jul-2018).

    Para cada timestamp repetido se conserva la vela cuyo cierre es más coherente
    con el precio de las velas vecinas no duplicadas.
    """
    dup = df["date"].duplicated(keep=False)
    if not dup.any():
        return df
    clean = df[~dup].reset_index(drop=True)
    keep = []
    for t, g in df[dup].groupby("date"):
        pos = clean["date"].searchsorted(t)
        ref = clean["close"].iloc[max(pos - 3, 0): pos + 3].median()
        keep.append((np.log(g["close"] / ref).abs()).idxmin())
    removed = int(dup.sum()) - len(keep)
    print(
        f"Limpieza: {removed} velas duplicadas/corruptas eliminadas "
        f"({df.date[dup].nunique()} timestamps repetidos)."
    )
    return pd.concat([clean, df.loc[keep]]).sort_values("date").reset_index(drop=True)


def drop_incomplete_last_candle(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta la última vela si su volumen es menos de la mitad de la mediana reciente."""
    if len(df) > 21 and df["volume"].iloc[-1] < 0.5 * df["volume"].iloc[-21:-1].median():
        print(f"Aviso: última vela ({df.date.iloc[-1].date()}) incompleta; se descarta.")
        return df.iloc[:-1].reset_index(drop=True)
    return df
