# Crea el entorno virtual e instala dependencias (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Listo. Un CSV:  python lab_btc.py --data data\btc_1d_data_2018_to_2025.csv"
Write-Host "Las 4 resoluciones: python lab_btc.py --data data\btc_4h_data_2018_to_2025.csv --out resultados\4h --max-train 40000  (ver run_all.sh)"
