# ============================================
# PREDICTOR LOTERÍA SANTANDER (FIXED VERSION)
# ============================================

import os
import sys
import warnings
from datetime import datetime, timedelta
from collections import Counter
from itertools import product

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import EXCEL_FILE, OUTPUT_DIR, HIST_FILE

warnings.filterwarnings("ignore")

RUTA_MODELO = os.path.join(ROOT_DIR, "modelo_loteria.pkl")

# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────

def cargar_datos(ruta):
    df = pd.read_excel(ruta, sheet_name="Histórico", header=3)
    df.columns = ["Fecha","Sorteo","Numero","Serie","D1","D2","D3","D4"]

    df = df.dropna(subset=["Numero"])
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Numero"] = df["Numero"].astype(str).str.zfill(4)

    for i in range(1,5):
        df[f"D{i}"] = df["Numero"].str[i-1].astype(int)

    return df.sort_values("Fecha").reset_index(drop=True)

# ─────────────────────────────────────────────
# MATRIZ TRANSICIÓN
# ─────────────────────────────────────────────

def matriz_transicion(df, pos):
    matriz = np.zeros((10,10))
    vals = df[f"D{pos}"].values

    for i in range(1, len(vals)):
        matriz[vals[i-1]][vals[i]] += 1

    matriz = np.nan_to_num(matriz / (matriz.sum(axis=1, keepdims=True)+1e-9))
    return matriz

# ─────────────────────────────────────────────
# FEATURES (FIX CLAVE)
# ─────────────────────────────────────────────

def crear_features(df, ventana=10):
    rows = []

    for idx in range(ventana, len(df)):
        hist = df.iloc[idx-ventana:idx]
        target = df.iloc[idx]

        feat = {}

        for pos in range(1,5):  # 🔥 YA CORRECTO
            col = f"D{pos}"
            vals = hist[col].values
            cnt = Counter(vals)

            mat = matriz_transicion(df.iloc[:idx], pos)
            ultimo = vals[-1]

            # transición
            for d in range(10):
                feat[f"trans_p{pos}_{d}"] = mat[ultimo][d]

            # frecuencia
            for d in range(10):
                feat[f"frec_p{pos}_{d}"] = cnt.get(d,0)/ventana

            # básicos
            feat[f"ult_p{pos}"] = vals[-1]
            feat[f"ult2_p{pos}"] = vals[-2]
            feat[f"media_p{pos}"] = np.mean(vals)

        # targets
        for p in range(1,5):
            feat[f"D{p}_target"] = target[f"D{p}"]

        rows.append(feat)

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────

def entrenar_modelos(feat_df):
    feature_cols = [c for c in feat_df.columns if "target" not in c]
    resultados = {}

    for pos in range(1,5):
        X = feat_df[feature_cols].fillna(0)
        y = feat_df[f"D{pos}_target"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=300, max_depth=12)
        rf.fit(X_scaled, y)

        xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
        xgb_m.fit(X_scaled, y)

        resultados[pos] = {
            "rf": rf,
            "xgb": xgb_m,
            "scaler": scaler,
            "X_cols": feature_cols,
            "cv_rf": cross_val_score(rf, X_scaled, y, cv=5).mean(),
            "cv_xgb": cross_val_score(xgb_m, X_scaled, y, cv=5).mean(),
        }

    joblib.dump(resultados, RUTA_MODELO)
    print("💾 Modelo guardado")

    return resultados

# ─────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────

def predecir(df, modelos):
    hist = df.tail(10)
    feat = {}

    for pos in range(1,5):
        vals = hist[f"D{pos}"].values
        cnt = Counter(vals)

        for d in range(10):
            feat[f"frec_p{pos}_{d}"] = cnt.get(d,0)/10

        feat[f"ult_p{pos}"] = vals[-1]
        feat[f"ult2_p{pos}"] = vals[-2]
        feat[f"media_p{pos}"] = np.mean(vals)

    X = pd.DataFrame([feat])

    for col in modelos[1]["X_cols"]:
        if col not in X:
            X[col] = 0

    X = X[modelos[1]["X_cols"]]

    numero = ""

    for pos in range(1,5):
        scaler = modelos[pos]["scaler"]
        X_scaled = scaler.transform(X)

        probs = modelos[pos]["xgb"].predict_proba(X_scaled)[0]
        numero += str(np.argmax(probs))

    return numero

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n🚀 Iniciando predictor...\n")

    df = cargar_datos(EXCEL_FILE)

    print(f"📊 Datos: {len(df)} sorteos")

    feat_df = crear_features(df)

    print(f"🧠 Samples entrenamiento: {len(feat_df)}")

    if os.path.exists(RUTA_MODELO):
        modelos = joblib.load(RUTA_MODELO)
        print("⚡ Modelo cargado")
    else:
        modelos = entrenar_modelos(feat_df)

    numero = predecir(df, modelos)

    print("\n🎯 NÚMERO PREDICHO:", numero)


if __name__ == "__main__":
    main()