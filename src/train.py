"""Entrena las tres tareas del laboratorio y deja cada intento como un run de MLflow."""

import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay

from config import (
    EXPERIMENTO,
    EXPERIMENTO_MAGNITUD,
    EXPERIMENTO_REGIMENES,
    SEMILLA,
    usar_almacen,
)
from data import cargar_datos
from btc_lab.models.catalog import default_classifiers, default_regressors
from btc_lab.tasks.clustering import ClusteringTask


def _guardar_modelo(modelo, X, pred):
    return mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        signature=infer_signature(X, pred),
        input_example=X.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )


def _clasificar(split, n_estimators: int):
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO)
    catalogo = default_classifiers(n_estimators, n_estimators, SEMILLA)
    ytr = split.Ttr.y_up.astype(int)
    yte = split.Tte.y_up.astype(int)
    for spec in catalogo.build():
        modelo = spec.build()
        with mlflow.start_run(run_name=spec.name):
            mlflow.log_params({"n_estimators": n_estimators, "seed": SEMILLA, "modelo": spec.name})
            mlflow.set_tags({"dataset": "bitcoin-1d", "tarea": "clasificacion", "baseline": str(spec.baseline).lower()})
            modelo.fit(split.Xtr, ytr)
            pred = modelo.predict(split.Xte)
            proba = modelo.predict_proba(split.Xte)[:, 1]
            metricas = {
                "accuracy": accuracy_score(yte, pred),
                "balanced_acc": balanced_accuracy_score(yte, pred),
                "f1": f1_score(yte, pred, zero_division=0),
                "roc_auc_test": roc_auc_score(yte, proba) if len(set(proba)) > 1 else 0.5,
            }
            mlflow.log_metrics(metricas)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["Baja", "Sube"], ax=ax)
            mlflow.log_figure(fig, "matriz_confusion.png")
            plt.close(fig)
            if not spec.baseline:
                info = _guardar_modelo(modelo, split.Xtr, modelo.predict(split.Xtr))
                mlflow.set_tag("model_uri", info.model_uri)
            print(f"clasificacion {spec.name}: accuracy={metricas['accuracy']:.3f} auc={metricas['roc_auc_test']:.3f}")


def _regresion(split, n_estimators: int):
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO_MAGNITUD)
    catalogo = default_regressors(n_estimators, n_estimators, SEMILLA)
    for spec in catalogo.build():
        modelo = spec.build()
        with mlflow.start_run(run_name=spec.name):
            mlflow.log_params({"n_estimators": n_estimators, "seed": SEMILLA, "modelo": spec.name, "target": "next_abs"})
            mlflow.set_tags({"dataset": "bitcoin-1d", "tarea": "magnitud", "baseline": str(spec.baseline).lower()})
            modelo.fit(split.Xtr, split.Ttr.next_abs)
            pred = modelo.predict(split.Xte)
            metricas = {
                "mae_pct": mean_absolute_error(split.Tte.next_abs, pred) * 100,
                "r2": r2_score(split.Tte.next_abs, pred),
            }
            mlflow.log_metrics(metricas)
            if not spec.baseline:
                info = _guardar_modelo(modelo, split.Xtr, modelo.predict(split.Xtr))
                mlflow.set_tag("model_uri", info.model_uri)
            print(f"magnitud {spec.name}: MAE={metricas['mae_pct']:.3f}% R2={metricas['r2']:.3f}")


def _agrupar(split):
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO_REGIMENES)
    with mlflow.start_run(run_name="kmeans"):
        mlflow.log_params({"seed": SEMILLA, "modelo": "KMeans"})
        mlflow.set_tags({"dataset": "bitcoin-1d", "tarea": "agrupacion", "baseline": "false"})
        with tempfile.TemporaryDirectory() as tmp:
            prof, k_best, sil = ClusteringTask(seed=SEMILLA).run(
                split.X, split.T, split.dates, split.close, tmp, None,
            )
            mlflow.log_metrics({"k": k_best, "silhouette": sil})
            mlflow.log_artifact(os.path.join(tmp, "agrupacion_perfiles.csv"))
            for nombre in ("agrupacion_k.png", "agrupacion_pca.png", "agrupacion_precio.png"):
                mlflow.log_artifact(os.path.join(tmp, nombre))
        print(f"agrupacion: k={k_best} silhouette={sil:.3f}")
        print(prof.to_string())


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de Bitcoin con seguimiento en MLflow")
    parser.add_argument("--data", default=None, help="CSV de velas. Por defecto data/btc_1d_data_2018_to_2025.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=60000)
    args = parser.parse_args()

    usar_almacen()
    _velas, split = cargar_datos(args.data, args.test_size, args.max_train)
    print(f"Train {len(split.Xtr)} | Test {len(split.Xte)} | columnas {split.Xtr.shape[1]}")
    _clasificar(split, args.n_estimators)
    _regresion(split, args.n_estimators)
    _agrupar(split)


if __name__ == "__main__":
    main()
