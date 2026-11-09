"""Corte temporal del histórico. En series de tiempo el futuro no entra al entrenamiento."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalSplit:
    X_all: pd.DataFrame
    X: pd.DataFrame
    T: pd.DataFrame
    Xtr: pd.DataFrame
    Xte: pd.DataFrame
    Ttr: pd.DataFrame
    Tte: pd.DataFrame
    dates: pd.Series
    close: pd.Series
    n_train_before_cap: int


def temporal_split(df: pd.DataFrame, X_all: pd.DataFrame, T_all: pd.DataFrame,
                   test_size: float, max_train: int) -> TemporalSplit:
    """Filtra filas completas, indexa por fecha y separa train/test en orden cronológico."""
    valid = X_all.notna().all(axis=1) & T_all.notna().all(axis=1)
    X, T = X_all[valid], T_all[valid]
    X.index = T.index = df.loc[X.index, "date"]
    dates = pd.Series(df["date"].values, index=df["date"])
    close = df.set_index("date")["close"]

    cut = int(len(X) * (1 - test_size))
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    Ttr, Tte = T.iloc[:cut], T.iloc[cut:]
    n_before = len(Xtr)
    if len(Xtr) > max_train:
        Xtr, Ttr = Xtr.iloc[-max_train:], Ttr.iloc[-max_train:]
    return TemporalSplit(
        X_all=X_all, X=X, T=T,
        Xtr=Xtr, Xte=Xte, Ttr=Ttr, Tte=Tte,
        dates=dates, close=close,
        n_train_before_cap=n_before,
    )
