"""Agrupación: regímenes de mercado con K-Means, sin usar el target para formar los grupos."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from btc_lab.reporting.plots import save_cluster_pca, save_cluster_price, save_cluster_selection

DEFAULT_COLUMNS = ("ret_5", "vol_10", "vol_20", "range", "vol_z", "sma_ratio_20", "rsi")


class ClusteringTask:
    def __init__(self, columns: tuple[str, ...] = DEFAULT_COLUMNS, seed: int = 42):
        self.columns = columns
        self.seed = seed

    def run(self, X, t, df_dates, close, outdir: str, k_fixed: int | None = None):
        cols = [c for c in self.columns if c in X.columns]
        Z = StandardScaler().fit_transform(X[cols])
        ks = list(range(2, 9))
        inertias, sils = [], []
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(Z)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(
                Z, km.labels_, sample_size=min(2000, len(Z)), random_state=self.seed,
            ))
        k_best = k_fixed or ks[int(np.argmax(sils))]
        km = KMeans(n_clusters=k_best, n_init=10, random_state=self.seed).fit(Z)
        labels = pd.Series(km.labels_, index=X.index, name="cluster")

        save_cluster_selection(ks, inertias, sils, f"{outdir}/agrupacion_k.png")
        projected = PCA(n_components=2, random_state=self.seed).fit_transform(Z)
        save_cluster_pca(projected, labels, k_best, f"{outdir}/agrupacion_pca.png")
        save_cluster_price(df_dates.loc[X.index], close.loc[X.index], labels, f"{outdir}/agrupacion_precio.png")

        prof = X[cols].groupby(labels).mean()
        prof["n_dias"] = labels.value_counts().sort_index()
        tt = t.loc[X.index].dropna()
        prof["mañana_|ret|_%"] = (tt["next_abs"] * 100).groupby(labels).mean()
        prof["mañana_%_sube"] = (tt["y_up"] * 100).groupby(labels).mean()
        prof = prof.round(4)
        prof.to_csv(f"{outdir}/agrupacion_perfiles.csv")
        return prof, k_best, max(sils)
