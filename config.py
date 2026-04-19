import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📂 Carpetas principales
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

# 📄 Archivos
EXCEL_FILE = os.path.join(DATA_DIR, "loteria_santander_v2.xlsx")
HIST_FILE  = os.path.join(OUTPUT_DIR, "predicciones_historial.xlsx")

# Crear carpetas automáticamente
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
