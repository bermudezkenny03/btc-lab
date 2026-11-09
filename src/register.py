"""Registra el mejor clasificador del experimento como 'challenger'."""

import mlflow
from mlflow import MlflowClient

from config import EXPERIMENTO, MODELO_REGISTRADO, usar_almacen

usar_almacen()
mejores = mlflow.search_runs(
    experiment_names=[EXPERIMENTO],
    filter_string="tags.tarea = 'clasificacion' and tags.baseline = 'false'",
    order_by=["metrics.roc_auc_test DESC"],
    max_results=1,
)
if mejores.empty:
    raise SystemExit(f"No hay runs en el experimento '{EXPERIMENTO}'. Ejecuta antes src/train.py")

mejor = mejores.iloc[0]
uri = mejor.get("tags.model_uri")
if not isinstance(uri, str) or not uri:
    raise SystemExit("El mejor run no tiene tags.model_uri. Vuelve a entrenar.")

version = mlflow.register_model(uri, MODELO_REGISTRADO)
MlflowClient().set_registered_model_alias(MODELO_REGISTRADO, "challenger", version.version)
print(
    f"Registrada la versión {version.version} "
    f"({mejor['tags.mlflow.runName']}, auc={mejor['metrics.roc_auc_test']:.3f}) como 'challenger'"
)
