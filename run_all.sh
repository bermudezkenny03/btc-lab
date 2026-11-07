#!/usr/bin/env bash
# Ejecuta el laboratorio en las 4 resoluciones y genera la comparación.
# Uso: source .venv/bin/activate && ./run_all.sh
set -e
D=data
run() { # $1=tf  $2..=args extra
  tf=$1; shift
  echo ">>> Resolución $tf"
  mkdir -p resultados/$tf
  python lab_btc.py --data $D/btc_${tf}_data_2018_to_2025.csv --out resultados/$tf "$@" | tee resultados/$tf/salida_consola.txt
}
run 1d
run 4h  --max-train 40000
run 1h  --max-train 20000 --fast
run 15m --max-train 20000 --fast
python compare_freqs.py
