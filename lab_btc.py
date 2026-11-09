"""Punto de entrada del laboratorio.

La lógica vive en el paquete ``btc_lab``. Este archivo solo lee los argumentos
y dispara una corrida, así ``python lab_btc.py`` y ``run_all.sh`` siguen igual.
"""

import argparse

from btc_lab.config import ExperimentConfig
from btc_lab.pipeline import run_experiment


def parse_args(argv=None) -> ExperimentConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Ruta al CSV diario (btc_1d_data_*.csv)")
    ap.add_argument("--synthetic", action="store_true", help="Usar datos sintéticos de prueba")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=None, help="Forzar número de clusters")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--max-train", type=int, default=60000,
                    help="Máx. filas de entrenamiento (se usan las más recientes) para acotar tiempo")
    ap.add_argument("--fast", action="store_true", help="Menos árboles (útil en 1h/15m)")
    args = ap.parse_args(argv)
    return ExperimentConfig(
        data_path=args.data,
        synthetic=args.synthetic,
        test_size=args.test_size,
        k=args.k,
        out=args.out,
        max_train=args.max_train,
        fast=args.fast,
    )


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
