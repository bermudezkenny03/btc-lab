"""Carga el clasificador registrado y predice si la siguiente vela sube."""

import argparse

import mlflow
import pandas as pd

from config import MODELO_REGISTRADO, usar_almacen
from data import cargar_velas
from btc_lab.features.builder import TechnicalFeatureBuilder

parser = argparse.ArgumentParser(description="Predicción con un modelo del registro")
parser.add_argument("csv", help="CSV de velas OHLCV, o de features ya calculadas")
parser.add_argument("--alias", default="challenger", help="alias del modelo (challenger, champion)")
args = parser.parse_args()

usar_almacen()
modelo = mlflow.pyfunc.load_model(f"models:/{MODELO_REGISTRADO}@{args.alias}")
columnas = [c.name for c in modelo.metadata.get_input_schema()]

muestra = pd.read_csv(args.csv, nrows=1)
nombres = {c.strip().lower().replace(" ", "_") for c in muestra.columns}
if {"open", "high", "low", "close", "volume"} <= nombres:
    velas = cargar_velas(args.csv)
    features, _targets = TechnicalFeatureBuilder().transform(velas)
    datos = features.dropna()
else:
    datos = pd.read_csv(args.csv)

faltan = [c for c in columnas if c not in datos.columns]
if faltan:
    raise SystemExit(f"Faltan columnas que el modelo espera: {faltan}")

entrada = datos[columnas]
pred = modelo.predict(entrada)
salida = datos.copy()
salida["prediccion_sube"] = pred
print(salida["prediccion_sube"].value_counts().sort_index().to_string())
ultima = salida.iloc[[-1]]
print(f"Última fila: prediccion_sube={int(ultima['prediccion_sube'].iloc[0])}")
salida.to_csv("predicciones.csv", index=False)
print("Guardado en predicciones.csv")
