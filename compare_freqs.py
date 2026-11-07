import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = "resultados"
rows = []
for tf in ("1d", "4h", "1h", "15m"):
    d = f"{R}/{tf}"
    if not os.path.exists(f"{d}/clasificacion_metricas.csv"):
        continue
    c = pd.read_csv(f"{d}/clasificacion_metricas.csv")
    base_c = c[c.modelo.str.startswith("Baseline")].iloc[0]
    best_c = c[~c.modelo.str.startswith("Baseline")].sort_values("roc_auc_test", ascending=False).iloc[0]
    m = pd.read_csv(f"{d}/regresion_next_abs_metricas.csv")
    base_m = m[m.modelo.str.startswith("Baseline")].iloc[0]
    best_m = m[~m.modelo.str.startswith("Baseline")].sort_values("MAE_%").iloc[0]
    r = pd.read_csv(f"{d}/regresion_next_ret_metricas.csv")
    best_dir = r[~r.modelo.str.startswith("Baseline")]["acierto_direccion"].max()
    k = pd.read_csv(f"{d}/agrupacion_perfiles.csv", index_col=0)
    big = k[k.n_dias >= 0.02 * k.n_dias.sum()]            # ignora clusters minúsculos (<2 %)
    ratio = big["mañana_|ret|_%"].max() / big["mañana_|ret|_%"].min()
    rows.append({
        "resolucion": tf,
        "mejor_clasificador": best_c.modelo,
        "accuracy": best_c.accuracy, "accuracy_baseline": base_c.accuracy,
        "roc_auc_test": best_c.roc_auc_test,
        "mejor_reg_magnitud": best_m.modelo,
        "MAE_%": best_m["MAE_%"], "MAE_baseline_%": base_m["MAE_%"],
        "mejora_MAE_%": 100 * (1 - best_m["MAE_%"] / base_m["MAE_%"]),
        "R2_magnitud": best_m.R2,
        "acierto_direccion_retorno": best_dir,
        "k_clusters": len(k), "ratio_riesgo_clusters": ratio,
    })
df = pd.DataFrame(rows).round(4)
df.to_csv(f"{R}/comparacion_resoluciones.csv", index=False)
print(df.T.to_string())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].bar(df.resolucion, df.roc_auc_test, color="#2a7ab0"); ax[0].axhline(0.5, color="r", ls="--")
ax[0].set_ylim(0.45, 0.6); ax[0].set_title("Clasificación: ROC AUC (test)\nlínea roja = azar")
ax[1].bar(df.resolucion, df["mejora_MAE_%"], color="#3a9a5b"); ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_title("Magnitud: mejora de MAE vs baseline (%)")
ax[2].bar(df.resolucion, df.ratio_riesgo_clusters, color="darkorange"); ax[2].axhline(1, color="k", lw=0.8)
ax[2].set_title("Agrupación: |ret| siguiente\n(cluster más volátil / más calmo)")
plt.tight_layout(); plt.savefig(f"{R}/comparacion_resoluciones.png", dpi=130)
