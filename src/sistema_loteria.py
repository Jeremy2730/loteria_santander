import os
import sys
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

script_actualizar = os.path.join(BASE_DIR, "src", "actualizar_loteria.py")
script_prediccion = os.path.join(BASE_DIR, "src", "prediccion_loteria.py")
script_evaluar    = os.path.join(BASE_DIR, "src", "evaluar_modelo.py")

print("="*60)
print("SISTEMA LOTERÍA SANTANDER")
print("Inicio:", datetime.now())
print("="*60)

print("\n1️⃣ Actualizando resultados...")
subprocess.run(
    [sys.executable, script_actualizar],
    cwd=BASE_DIR
)

print("\n2️⃣ Ejecutando predictor...")
subprocess.run(
    [sys.executable, script_prediccion],
    cwd=BASE_DIR
)

print("\n✅ Proceso terminado.")
print("Archivos generados en /output")

print("\n3️⃣ Evaluando rendimiento del modelo...")
subprocess.run(
    [sys.executable, script_evaluar],
    cwd=BASE_DIR
)