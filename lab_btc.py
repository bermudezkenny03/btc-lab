import argparse
import glob
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
N_RF, N_GB = 300, 150          # nº de árboles (--fast los reduce)
KAGGLE_ID = "novandraanugrah/bitcoin-historical-datasets-2018-2024"


# --------------------------------------------------------------------------- #
# 1. CARGA DE DATOS
# --------------------------------------------------------------------------- #
def find_csv(path_arg):
    """Devuelve la ruta del CSV diario (local o descargado con kagglehub)."""
    if path_arg:
        return path_arg
    local = sorted(glob.glob("*1d*.csv") + glob.glob("data/*1d*.csv"))
    if local:
        return local[0]
    import kagglehub
    folder = kagglehub.dataset_download(KAGGLE_ID)
    cands = sorted(glob.glob(os.path.join(folder, "*1d*.csv")))
    if not cands:
        raise FileNotFoundError(f"No encontré un CSV diario (*1d*.csv) en {folder}")
    return cands[0]


def synthetic_data(n=2500):
    """Serie sintética con volatilidad agrupada, solo para probar el pipeline."""
    rng = np.random.default_rng(SEED)
    vol = np.zeros(n); vol[0] = 0.03
    ret = np.zeros(n)
    for t in range(1, n):
        vol[t] = np.sqrt(1e-5 + 0.1 * ret[t-1]**2 + 0.85 * vol[t-1]**2)
        ret[t] = vol[t] * rng.standard_normal()
    close = 5000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol)))
    volume = 1e4 * np.exp(rng.normal(0, 0.3, n) + 8 * vol)
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "number_of_trades": (volume * 50).astype(int),
        "taker_buy_base_asset_volume": volume * rng.uniform(0.4, 0.6, n),
    })
    return df


def fix_duplicate_timestamps(df):
    """El dataset trae velas duplicadas con precios corruptos (~108 000 USD en jul-2018).
    Para cada timestamp repetido se conserva la vela cuyo cierre es más coherente con
    el precio de las velas vecinas no duplicadas."""
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
    print(f"Limpieza: {removed} velas duplicadas/corruptas eliminadas ({df.date[dup].nunique()} timestamps repetidos).")
    return pd.concat([clean, df.loc[keep]]).sort_values("date").reset_index(drop=True)


def load_data(path):
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
    df = fix_duplicate_timestamps(df)
    return df


# --------------------------------------------------------------------------- #
# 2. INGENIERÍA DE CARACTERÍSTICAS (solo información pasada: sin fuga de datos)
# --------------------------------------------------------------------------- #
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(df):
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
    f["dow_sin"], f["dow_cos"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    if df["date"].diff().median() < pd.Timedelta(days=1):      # datos intradía
        hr = df["date"].dt.hour + df["date"].dt.minute / 60
        f["hour_sin"], f["hour_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)

    # Targets (lo que ocurre MAÑANA respecto a hoy)
    nxt = lr.shift(-1)
    t = pd.DataFrame({"next_ret": nxt,
                      "y_up": (nxt > 0).astype(float).where(nxt.notna()),
                      "next_abs": nxt.abs()})
    return f.replace([np.inf, -np.inf], np.nan), t


# --------------------------------------------------------------------------- #
# 3. CLASIFICACIÓN
# --------------------------------------------------------------------------- #
def run_classification(Xtr, Xte, ytr, yte, outdir):
    models = {
        "Baseline (mayoría)": DummyClassifier(strategy="most_frequent"),
        "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)),
        "Random Forest": RandomForestClassifier(n_estimators=N_RF, min_samples_leaf=10,
                                                max_depth=6, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=N_GB, max_depth=2,
                                                        learning_rate=0.05, subsample=0.8,
                                                        random_state=SEED),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rows, fitted = [], {}
    for name, m in models.items():
        cv_auc = cross_val_score(m, Xtr, ytr, cv=tscv, scoring="roc_auc").mean() if "Baseline" not in name else 0.5
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        rows.append({"modelo": name,
                     "accuracy": accuracy_score(yte, pred),
                     "balanced_acc": balanced_accuracy_score(yte, pred),
                     "f1": f1_score(yte, pred, zero_division=0),
                     "roc_auc_test": roc_auc_score(yte, proba) if len(set(proba)) > 1 else 0.5,
                     "roc_auc_cv_train": cv_auc})
        fitted[name] = m
    res = pd.DataFrame(rows).round(4)
    res.to_csv(f"{outdir}/clasificacion_metricas.csv", index=False)

    best = res[res.modelo != "Baseline (mayoría)"].sort_values("roc_auc_test", ascending=False).iloc[0].modelo
    cm = confusion_matrix(yte, fitted[best].predict(Xte))
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Baja", "Sube"], yticklabels=["Baja", "Sube"])
    plt.xlabel("Predicho"); plt.ylabel("Real"); plt.title(f"Matriz de confusión\n{best}")
    plt.tight_layout(); plt.savefig(f"{outdir}/clasificacion_confusion.png", dpi=130); plt.close()

    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values().tail(12)
    plt.figure(figsize=(6, 4.5)); imp.plot.barh(color="#2a7ab0")
    plt.title("Importancia de variables (Random Forest) – ¿sube/baja?")
    plt.tight_layout(); plt.savefig(f"{outdir}/clasificacion_importancias.png", dpi=130); plt.close()
    return res, fitted, best


# --------------------------------------------------------------------------- #
# 4. REGRESIÓN
# --------------------------------------------------------------------------- #
def reg_models():
    return {
        "Baseline (media train)": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Random Forest": RandomForestRegressor(n_estimators=N_RF, min_samples_leaf=10,
                                               max_depth=6, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=N_GB, max_depth=2,
                                                       learning_rate=0.05, subsample=0.8,
                                                       random_state=SEED),
    }


def run_regression(Xtr, Xte, t_tr, t_te, outdir):
    out = {}
    for target, label in (("next_ret", "Retorno con signo"), ("next_abs", "Magnitud |retorno|")):
        rows, fitted = [], {}
        for name, m in reg_models().items():
            m.fit(Xtr, t_tr[target])
            p = m.predict(Xte)
            r = {"modelo": name,
                 "MAE_%": mean_absolute_error(t_te[target], p) * 100,
                 "RMSE_%": np.sqrt(mean_squared_error(t_te[target], p)) * 100,
                 "R2": r2_score(t_te[target], p)}
            if target == "next_ret":
                r["acierto_direccion"] = (np.sign(p) == np.sign(t_te[target])).mean()
            rows.append(r); fitted[name] = (m, p)
        res = pd.DataFrame(rows).round(4)
        res.to_csv(f"{outdir}/regresion_{target}_metricas.csv", index=False)
        out[target] = (res, fitted, label)

    # Gráfico: real vs predicho de la magnitud (ventana final del test)
    res, fitted, _ = out["next_abs"]
    best = res[res.modelo != "Baseline (media train)"].sort_values("R2", ascending=False).iloc[0].modelo
    pred = fitted[best][1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    idx = t_te.index
    ax[0].plot(idx, t_te["next_abs"] * 100, lw=0.8, label="Real", alpha=0.8)
    ax[0].plot(idx, pred * 100, lw=1.2, label=f"Predicho ({best})")
    ax[0].set_title("Magnitud del movimiento de mañana (%) – test"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=30)
    ax[1].scatter(t_te["next_abs"] * 100, pred * 100, s=8, alpha=0.5)
    lim = max(t_te["next_abs"].max(), pred.max()) * 100
    ax[1].plot([0, lim], [0, lim], "r--", lw=1)
    ax[1].set_xlabel("Real (%)"); ax[1].set_ylabel("Predicho (%)"); ax[1].set_title("Real vs predicho")
    plt.tight_layout(); plt.savefig(f"{outdir}/regresion_magnitud.png", dpi=130); plt.close()
    return out, best


# --------------------------------------------------------------------------- #
# 5. AGRUPACIÓN (K-Means) – regímenes de mercado
# --------------------------------------------------------------------------- #
def run_clustering(X, t, df_dates, close, outdir, k_fixed=None):
    cols = [c for c in ("ret_5", "vol_10", "vol_20", "range", "vol_z", "sma_ratio_20", "rsi") if c in X.columns]
    Z = StandardScaler().fit_transform(X[cols])
    ks, inertias, sils = list(range(2, 9)), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Z, km.labels_, sample_size=min(2000, len(Z)), random_state=SEED))
    k_best = k_fixed or ks[int(np.argmax(sils))]
    km = KMeans(n_clusters=k_best, n_init=10, random_state=SEED).fit(Z)
    labels = pd.Series(km.labels_, index=X.index, name="cluster")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ks, inertias, "o-"); ax[0].set_title("Codo (inercia)"); ax[0].set_xlabel("k")
    ax[1].plot(ks, sils, "o-", color="darkorange"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
    plt.tight_layout(); plt.savefig(f"{outdir}/agrupacion_k.png", dpi=130); plt.close()

    p = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=p[:, 0], y=p[:, 1], hue=labels, palette="tab10", s=14, alpha=0.7)
    plt.title(f"Regímenes de mercado (K-Means, k={k_best}) – PCA 2D")
    plt.tight_layout(); plt.savefig(f"{outdir}/agrupacion_pca.png", dpi=130); plt.close()

    plt.figure(figsize=(12, 4))
    dates = df_dates.loc[X.index]
    plt.scatter(dates, close.loc[X.index], c=labels, cmap="tab10", s=6)
    plt.yscale("log"); plt.title("Precio BTC (log) coloreado por régimen")
    plt.tight_layout(); plt.savefig(f"{outdir}/agrupacion_precio.png", dpi=130); plt.close()

    prof = X[cols].groupby(labels).mean()
    prof["n_dias"] = labels.value_counts().sort_index()
    tt = t.loc[X.index].dropna()
    prof["mañana_|ret|_%"] = (tt["next_abs"] * 100).groupby(labels).mean()
    prof["mañana_%_sube"] = (tt["y_up"] * 100).groupby(labels).mean()
    prof = prof.round(4)
    prof.to_csv(f"{outdir}/agrupacion_perfiles.csv")
    return prof, k_best, max(sils)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Ruta al CSV diario (btc_1d_data_*.csv)")
    ap.add_argument("--synthetic", action="store_true", help="Usar datos sintéticos de prueba")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=None, help="Forzar número de clusters")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--max-train", type=int, default=60000,
                    help="Máx. filas de entrenamiento (se usan las más recientes) para acotar tiempo")
    ap.add_argument("--fast", action="store_true", help="Menos árboles (útil en 1h/15m)")
    a = ap.parse_args()
    global N_RF, N_GB
    if a.fast:
        N_RF, N_GB = 100, 80
    os.makedirs(a.out, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = synthetic_data() if a.synthetic else load_data(find_csv(a.data))
    # Si la última vela es parcial (volumen muy bajo vs. días previos), se descarta
    if len(df) > 21 and df["volume"].iloc[-1] < 0.5 * df["volume"].iloc[-21:-1].median():
        print(f"Aviso: última vela ({df.date.iloc[-1].date()}) incompleta; se descarta.")
        df = df.iloc[:-1].reset_index(drop=True)
    print(f"Datos: {len(df)} filas | {df.date.min().date()} → {df.date.max().date()}")

    X_all, T_all = build_features(df)
    last_X = X_all.iloc[[-1]].dropna(axis=1, how="any")          # fila de HOY para pronosticar mañana
    valid = X_all.notna().all(axis=1) & T_all.notna().all(axis=1)
    X, T = X_all[valid], T_all[valid]
    X.index = T.index = df.loc[X.index, "date"]
    dates_s = pd.Series(df["date"].values, index=df["date"])
    close_s = df.set_index("date")["close"]

    # Split temporal (nunca aleatorio en series de tiempo)
    cut = int(len(X) * (1 - a.test_size))
    Xtr, Xte, Ttr, Tte = X.iloc[:cut], X.iloc[cut:], T.iloc[:cut], T.iloc[cut:]
    if len(Xtr) > a.max_train:
        print(f"Aviso: se usan las {a.max_train} filas de entrenamiento más recientes (de {len(Xtr)}).")
        Xtr, Ttr = Xtr.iloc[-a.max_train:], Ttr.iloc[-a.max_train:]
    print(f"Train: {len(Xtr)} ({Xtr.index.min().date()}→{Xtr.index.max().date()}) | "
          f"Test: {len(Xte)} ({Xte.index.min().date()}→{Xte.index.max().date()})")
    print(f"% de días que suben (train/test): {Ttr.y_up.mean():.1%} / {Tte.y_up.mean():.1%}")

    print("\n=== 1) CLASIFICACIÓN: ¿sube o baja mañana? ===")
    cls_res, cls_models, cls_best = run_classification(Xtr, Xte, Ttr.y_up.astype(int), Tte.y_up.astype(int), a.out)
    print(cls_res.to_string(index=False))

    print("\n=== 2) REGRESIÓN: ¿qué tan grande es el movimiento? ===")
    reg_out, reg_best = run_regression(Xtr, Xte, Ttr, Tte, a.out)
    for target, (res, _, label) in reg_out.items():
        print(f"\n-- {label} --"); print(res.to_string(index=False))

    print("\n=== 3) AGRUPACIÓN: regímenes de mercado (K-Means) ===")
    prof, k_best, sil = run_clustering(X, T, dates_s, close_s, a.out, a.k)
    print(f"k elegido = {k_best} (silhouette = {sil:.3f})")
    print(prof.to_string())

    # Pronóstico para mañana con los mejores modelos
    xl = X_all.iloc[[-1]][X.columns]
    if xl.notna().all(axis=1).iloc[0]:
        # reentrena con TODO el histórico válido para pronosticar mañana
        c_model = cls_models[cls_best]; c_model.fit(X, T.y_up.astype(int))
        r_model = reg_models()[reg_best]; r_model.fit(X, T.next_abs)
        p_up = c_model.predict_proba(xl)[0, 1]
        mag = r_model.predict(xl)[0] * 100
        fc = {"fecha_base": str(df.date.iloc[-1]), "prob_sube": round(float(p_up), 4),
              "magnitud_esperada_pct": round(float(mag), 3),
              "modelo_clasificacion": cls_best, "modelo_magnitud": reg_best}
        json.dump(fc, open(f"{a.out}/pronostico_manana.json", "w"), indent=2, ensure_ascii=False)
        print(f"\nPronóstico para la siguiente vela tras {fc['fecha_base']}: P(sube)={p_up:.1%} | movimiento esperado ≈ ±{mag:.2f}%")

    print(f"\nArchivos guardados en {a.out}/")


if __name__ == "__main__":
    main()
