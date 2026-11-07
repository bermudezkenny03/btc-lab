#!/usr/bin/env bash
# Crea el entorno virtual e instala dependencias (Linux / macOS / WSL / Git Bash)
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Listo. Activa el entorno con:  source .venv/bin/activate"
echo "Un solo CSV:  python lab_btc.py --data data/btc_1d_data_2018_to_2025.csv"
echo "Las 4 resoluciones + comparación:  ./run_all.sh"
