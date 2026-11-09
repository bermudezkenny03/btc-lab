"""Registra en MLflow las mismas tareas que ya calcula el paquete btc_lab."""

import argparse
import os
import tempfile

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from config import (
    EXPERIMENTO,
    EXPERIMENTO_MAGNITUD,
    EXPERIMENTO_PRONOSTICO,
    EXPERIMENTO_REGIMENES,
    EXPERIMENTO_RETORNO,
    SEMILLA,
    usar_almacen,
)
from data import cargar_datos
from btc_lab.config import ExperimentConfig
from btc_lab.models.catalog import default_classifiers, default_regressors
from btc_lab.tasks.classification import ClassificationTask
from btc_lab.tasks.clustering import ClusteringTask
from btc_lab.tasks.forecast import ForecastTask
from btc_lab.tasks.regression import RegressionTask


def _guardar_modelo(modelo, X):
    pred = modelo.predict(X)
    return mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        signature=infer_signature(X, pred),
        input_example=X.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )


def _abrir(nombre_experimento: str):
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(nombre_experimento)


def _artefacto(carpeta: str, nombre: str):
    ruta = os.path.join(carpeta, nombre)
    if os.path.exists(ruta):
        mlflow.log_artifact(ruta)


def _clasificar(split, n_rf: int, n_gb: int):
    _abrir(EXPERIMENTO)
    catalogo = default_classifiers(n_rf, n_gb, SEMILLA)
    ytr = split.Ttr.y_up.astype(int)
    yte = split.Tte.y_up.astype(int)
    with tempfile.TemporaryDirectory() as tmp:
        res, fitted, best = ClassificationTask(catalogo).run(split.Xtr, split.Xte, ytr, yte, tmp)
        for _, row in res.iterrows():
            nombre = row.modelo
            baseline = nombre in catalogo.baseline_names
            with mlflow.start_run(run_name=nombre):
                mlflow.log_params({
                    "n_estimators_rf": n_rf,
                    "n_estimators_gb": n_gb,
                    "seed": SEMILLA,
                    "modelo": nombre,
                })
                mlflow.set_tags({
                    "dataset": "bitcoin-1d",
                    "tarea": "clasificacion",
                    "baseline": str(baseline).lower(),
                })
                mlflow.log_metrics({
                    "accuracy": float(row.accuracy),
                    "balanced_acc": float(row.balanced_acc),
                    "f1": float(row.f1),
                    "roc_auc_test": float(row.roc_auc_test),
                    "roc_auc_cv_train": float(row.roc_auc_cv_train),
                })
                _artefacto(tmp, "clasificacion_metricas.csv")
                if nombre == best:
                    _artefacto(tmp, "clasificacion_confusion.png")
                if nombre == "Random Forest":
                    _artefacto(tmp, "clasificacion_importancias.png")
                if not baseline:
                    info = _guardar_modelo(fitted[nombre], split.Xtr)
                    mlflow.set_tag("model_uri", info.model_uri)
            print(f"clasificacion {nombre}: accuracy={row.accuracy:.3f} auc={row.roc_auc_test:.3f} cv={row.roc_auc_cv_train:.3f}")
    return fitted, best


def _regresion(split, n_rf: int, n_gb: int):
    catalogo = default_regressors(n_rf, n_gb, SEMILLA)
    experimentos = {"next_abs": EXPERIMENTO_MAGNITUD, "next_ret": EXPERIMENTO_RETORNO}
    with tempfile.TemporaryDirectory() as tmp:
        salida, best = RegressionTask(catalogo).run(split.Xtr, split.Xte, split.Ttr, split.Tte, tmp)
        for target, (res, fitted, label) in salida.items():
            _abrir(experimentos[target])
            for _, row in res.iterrows():
                nombre = row.modelo
                baseline = nombre in catalogo.baseline_names
                with mlflow.start_run(run_name=nombre):
                    mlflow.log_params({
                        "n_estimators_rf": n_rf,
                        "n_estimators_gb": n_gb,
                        "seed": SEMILLA,
                        "modelo": nombre,
                        "target": target,
                    })
                    mlflow.set_tags({
                        "dataset": "bitcoin-1d",
                        "tarea": "regresion",
                        "baseline": str(baseline).lower(),
                    })
                    metricas = {
                        "mae_pct": float(row["MAE_%"]),
                        "rmse_pct": float(row["RMSE_%"]),
                        "r2": float(row["R2"]),
                    }
                    if "acierto_direccion" in row and row["acierto_direccion"] == row["acierto_direccion"]:
                        metricas["acierto_direccion"] = float(row["acierto_direccion"])
                    mlflow.log_metrics(metricas)
                    _artefacto(tmp, f"regresion_{target}_metricas.csv")
                    if target == "next_abs" and nombre == best:
                        _artefacto(tmp, "regresion_magnitud.png")
                    if not baseline:
                        info = _guardar_modelo(fitted[nombre][0], split.Xtr)
                        mlflow.set_tag("model_uri", info.model_uri)
                print(f"{label} {nombre}: MAE={row['MAE_%']:.3f}% R2={row['R2']:.3f}")
    return catalogo, best


def _pronostico(velas, split, cls_models, cls_best, regresores, reg_best):
    _abrir(EXPERIMENTO_PRONOSTICO)
    with tempfile.TemporaryDirectory() as tmp:
        forecast = ForecastTask(regresores).run(
            velas, split.X, split.T, split.X_all, cls_models, cls_best, reg_best, tmp,
        )
        if forecast is None:
            print("Pronóstico: la última fila no tiene features completas.")
            return
        with mlflow.start_run(run_name="siguiente-vela"):
            mlflow.log_params({
                "modelo_clasificacion": forecast.modelo_clasificacion,
                "modelo_magnitud": forecast.modelo_magnitud,
                "seed": SEMILLA,
            })
            mlflow.set_tags({"dataset": "bitcoin-1d", "tarea": "pronostico", "baseline": "false"})
            mlflow.log_metrics({
                "prob_sube": forecast.prob_sube,
                "magnitud_pct": forecast.magnitud_pct,
            })
            _artefacto(tmp, "pronostico_manana.json")
        print(
            f"Pronóstico tras {forecast.fecha_base}: "
            f"P(sube)={forecast.prob_sube:.1%} | movimiento ≈ ±{forecast.magnitud_pct:.2f}%"
        )


def _agrupar(split):
    _abrir(EXPERIMENTO_REGIMENES)
    with mlflow.start_run(run_name="kmeans"):
        mlflow.log_params({"seed": SEMILLA, "modelo": "KMeans"})
        mlflow.set_tags({"dataset": "bitcoin-1d", "tarea": "agrupacion", "baseline": "false"})
        with tempfile.TemporaryDirectory() as tmp:
            prof, k_best, sil = ClusteringTask(seed=SEMILLA).run(
                split.X, split.T, split.dates, split.close, tmp, None,
            )
            mlflow.log_metrics({"k": float(k_best), "silhouette": float(sil)})
            for nombre in (
                "agrupacion_perfiles.csv",
                "agrupacion_k.png",
                "agrupacion_pca.png",
                "agrupacion_precio.png",
            ):
                _artefacto(tmp, nombre)
        print(f"agrupacion: k={k_best} silhouette={sil:.3f}")
        print(prof.to_string())


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de Bitcoin con seguimiento en MLflow")
    parser.add_argument("--data", default=None, help="CSV de velas. Por defecto data/btc_1d_data_2018_to_2025.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=60000)
    parser.add_argument("--fast", action="store_true", help="Menos árboles, igual que lab_btc.py --fast")
    args = parser.parse_args()

    ajustes = ExperimentConfig(fast=args.fast, seed=SEMILLA)
    usar_almacen()
    velas, split = cargar_datos(args.data, args.test_size, args.max_train)
    print(
        f"Train {len(split.Xtr)} | Test {len(split.Xte)} | columnas {split.Xtr.shape[1]} | "
        f"RF {ajustes.n_estimators_rf} árboles, GB {ajustes.n_estimators_gb}"
    )
    cls_models, cls_best = _clasificar(split, ajustes.n_estimators_rf, ajustes.n_estimators_gb)
    regresores, reg_best = _regresion(split, ajustes.n_estimators_rf, ajustes.n_estimators_gb)
    _pronostico(velas, split, cls_models, cls_best, regresores, reg_best)
    _agrupar(split)


if __name__ == "__main__":
    main()
