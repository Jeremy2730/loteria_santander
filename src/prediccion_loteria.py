"""
=============================================================
  PREDICTOR LOTERÍA DE SANTANDER
  Modelos: Random Forest + XGBoost + Frecuencia ponderada
  Genera gráficos PNG y pronóstico semanal
=============================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings, os, sys
from datetime import datetime, timedelta
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import EXCEL_FILE, OUTPUT_DIR, HIST_FILE

plt.ioff()

warnings.filterwarnings("ignore")

# ── CONFIGURACIÓN ────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

RUTA_EXCEL = EXCEL_FILE
CARPETA_SALIDA = OUTPUT_DIR

os.makedirs(CARPETA_SALIDA, exist_ok=True)

COLORES = {
    "azul_oscuro": "#1F4E79",
    "azul_medio":  "#2E75B6",
    "azul_claro":  "#BDD7EE",
    "naranja":     "#C55A11",
    "verde":       "#538135",
    "rojo":        "#C00000",
    "gris":        "#595959",
    "fondo":       "#F5F8FC",
}
# ──────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════
#  1. CARGA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════

def analizar_resultado(predicho, real):
    # 🔥 asegurar strings SIEMPRE
    predicho = str(predicho).zfill(4)
    real = str(real).zfill(4)

    detalle = []
    aciertos = 0

    for p, r in zip(predicho, real):
        if p == r:
            detalle.append("✔")
            aciertos += 1
        else:
            detalle.append("✖")

    return aciertos, "".join(detalle)


def cargar_datos(ruta):
    df = pd.read_excel(ruta, sheet_name="Histórico", header=3)
    df.columns = ["Fecha","Sorteo","Numero","Serie","D1","D2","D3","D4"]
    df = df.dropna(subset=["Numero"])
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Numero"] = df["Numero"].astype(str).str.zfill(4)
    for i in range(1, 5):
        df[f"D{i}"] = df["Numero"].str[i-1].astype(int)
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df

# ═══════════════════════════════════════════════════════════
#  2. INGENIERÍA DE CARACTERÍSTICAS
# ═══════════════════════════════════════════════════════════

def crear_features(df, ventana=10):
    rows = []

    for idx in range(ventana, len(df)):
        hist = df.iloc[idx - ventana:idx]
        target = df.iloc[idx]
        feat = {}

        for pos in range(1, 5):
            col = f"D{pos}"
            vals = hist[col].values
            cnt  = Counter(vals)

            for d in range(10):
                feat[f"frec_p{pos}_{d}"] = cnt.get(d, 0) / ventana

            feat[f"ult_p{pos}"]  = vals[-1]
            feat[f"ult2_p{pos}"] = vals[-2]
            feat[f"media_p{pos}"] = np.mean(vals)
            feat[f"std_p{pos}"]   = np.std(vals)

            ultimo_dig = target[col]
            ss = 0
            for v in reversed(vals):
                if v == ultimo_dig:
                    break
                ss += 1
            feat[f"ss_p{pos}"] = ss

            feat[f"cambio_p{pos}"] = vals[-1] - vals[-2]
            feat[f"tendencia_p{pos}"] = np.mean(np.diff(vals))
            feat[f"max_p{pos}"] = np.max(vals)
            feat[f"min_p{pos}"] = np.min(vals)

        # 🔥 TARGETS (OBLIGATORIO)
        feat["D1_target"] = target["D1"]
        feat["D2_target"] = target["D2"]
        feat["D3_target"] = target["D3"]
        feat["D4_target"] = target["D4"]

        feat["semana"] = target["Fecha"].isocalendar()[1]
        feat["mes"]    = target["Fecha"].month

        rows.append(feat)

    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════
#  3. ENTRENAMIENTO DE MODELOS
# ═══════════════════════════════════════════════════════════

def entrenar_modelos(feat_df):
    feature_cols = [c for c in feat_df.columns if "target" not in c]
    resultados = {}

    for pos in range(1, 5):
        y = feat_df[f"D{pos}_target"].astype(int)
        X = feat_df[feature_cols].fillna(0)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                    random_state=42, n_jobs=-1)
        rf.fit(X, y)
        cv_rf = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()

        # XGBoost
        xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=5,
                                   learning_rate=0.05, use_label_encoder=False,
                                   eval_metric="mlogloss", random_state=42,
                                   verbosity=0)
        xgb_m.fit(X, y)
        cv_xgb = cross_val_score(xgb_m, X, y, cv=5, scoring="accuracy").mean()

        resultados[pos] = {
            "rf": rf,
            "xgb": xgb_m,
            "scaler": scaler,   # 🔥 AÑADE ESTA LÍNEA
            "cv_rf": cv_rf,
            "cv_xgb": cv_xgb,
            "X_cols": feature_cols,
        }
        print(f"  Posición {pos} → RF: {cv_rf:.1%}  XGB: {cv_xgb:.1%}")

    return resultados

# ═══════════════════════════════════════════════════════════
#  4. PREDICCIÓN PARA UNA FECHA
# ═══════════════════════════════════════════════════════════

def predecir_para_fecha(df, modelos, fecha_objetivo=None, ventana=10, historial=None):

    if fecha_objetivo is None:
        hoy = datetime.today()
        dias = (4 - hoy.weekday()) % 7
        if dias == 0 and hoy.hour >= 23:
            dias = 7
        fecha_objetivo = hoy + timedelta(days=dias if dias > 0 else 7)

    hist = df.tail(ventana)

    feat = {}

    for pos in range(1, 5):
        col  = f"D{pos}"
        vals = hist[col].values
        cnt  = Counter(vals)

        # 🔹 FRECUENCIAS
        for d in range(10):
            feat[f"frec_p{pos}_{d}"] = cnt.get(d, 0) / ventana

        # 🔹 BÁSICAS
        feat[f"ult_p{pos}"]  = vals[-1]
        feat[f"ult2_p{pos}"] = vals[-2]
        feat[f"media_p{pos}"] = np.mean(vals)
        feat[f"std_p{pos}"]   = np.std(vals)

        # 🔹 SECUENCIA SIN SALIR
        ultimo = vals[-1]
        ss = 0
        for v in reversed(vals[:-1]):
            if v == ultimo:
                break
            ss += 1
        feat[f"ss_p{pos}"] = ss

        # 🔥 🔥 🔥 AQUÍ ESTABA EL ERROR 🔥 🔥 🔥
        feat[f"cambio_p{pos}"] = vals[-1] - vals[-2]

        if len(vals) > 2:
            feat[f"tendencia_p{pos}"] = np.mean(np.diff(vals))
        else:
            feat[f"tendencia_p{pos}"] = 0

        feat[f"max_p{pos}"] = np.max(vals)
        feat[f"min_p{pos}"] = np.min(vals)

    # 🔹 FECHA
    feat["semana"] = fecha_objetivo.isocalendar()[1]
    feat["mes"]    = fecha_objetivo.month

    # 🔥 CLAVE: asegurar mismas columnas
    X_pred = pd.DataFrame([feat])

    for col in modelos[1]["X_cols"]:
        if col not in X_pred.columns:
            X_pred[col] = 0

    X_pred = X_pred.reindex(columns=modelos[1]["X_cols"], fill_value=0)

    probabilidades = {}

    for pos in range(1, 5):
        scaler = modelos[pos]["scaler"]
        X_scaled = scaler.transform(X_pred)

        rf_probs  = modelos[pos]["rf"].predict_proba(X_scaled)[0]
        xgb_probs = modelos[pos]["xgb"].predict_proba(X_scaled)[0]

        clases_rf  = modelos[pos]["rf"].classes_
        clases_xgb = modelos[pos]["xgb"].classes_

        prob_vec = np.zeros(10)

        for i, c in enumerate(clases_rf):
            prob_vec[c] += rf_probs[i] * 0.5

        for i, c in enumerate(clases_xgb):
            prob_vec[c] += xgb_probs[i] * 0.5

        # 🔹 HISTÓRICO GLOBAL
        hist_global = Counter(df[f"D{pos}"].values)
        total_hist  = sum(hist_global.values())

        hist_vec = np.array([hist_global.get(d, 0)/total_hist for d in range(10)])

        # 🔹 PESO DINÁMICO
        def calcular_peso(historial):
            if historial is None or len(historial) < 10:
                return 0.8

            ultimos = historial.tail(20)

            promedio = ultimos["aciertos"].mean()
            estabilidad = 1 - ultimos["aciertos"].std() / 2 if len(ultimos) > 1 else 0.5

            score = (promedio / 4) * 0.7 + estabilidad * 0.3
            peso = 0.4 + score * 0.5

            return min(max(peso, 0.4), 0.9)

        peso_modelo = calcular_peso(historial)
        peso_hist   = 1 - peso_modelo

        prob_final = peso_modelo * prob_vec + peso_hist * hist_vec

        # 🔹 ruido leve
        prob_final = np.clip(prob_final, 1e-6, None)

        suma = prob_final.sum()

        if suma <= 0 or np.isnan(suma):
            prob_final = np.ones(10) / 10
        else:
            prob_final = prob_final / suma

        probabilidades[pos] = prob_final

    return probabilidades, fecha_objetivo

# ═══════════════════════════════════════════════════════════
#  5. GRÁFICOS
# ═══════════════════════════════════════════════════════════

def grafico_frecuencias_historicas(df, ruta_salida):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(COLORES["fondo"])
    fig.suptitle("FRECUENCIA HISTÓRICA POR POSICIÓN\nLotería de Santander",
                 fontsize=16, fontweight="bold", color=COLORES["azul_oscuro"], y=0.98)

    pos_names = ["1ª Cifra", "2ª Cifra", "3ª Cifra", "4ª Cifra"]
    for idx, (ax, pname) in enumerate(zip(axes.flat, pos_names)):
        pos = idx + 1
        cnt = Counter(df[f"D{pos}"].values)
        digitos = list(range(10))
        frecuencias = [cnt.get(d, 0) for d in digitos]
        max_f = max(frecuencias)
        colores_bar = [COLORES["azul_oscuro"] if f == max_f
                       else COLORES["azul_medio"] if f >= np.percentile(frecuencias, 70)
                       else COLORES["azul_claro"] for f in frecuencias]

        bars = ax.bar(digitos, frecuencias, color=colores_bar,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_facecolor(COLORES["fondo"])
        ax.set_title(pname, fontsize=13, fontweight="bold",
                     color=COLORES["azul_oscuro"])
        ax.set_xlabel("Dígito", fontsize=10, color=COLORES["gris"])
        ax.set_ylabel("Frecuencia", fontsize=10, color=COLORES["gris"])
        ax.set_xticks(digitos)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        linea_esperada = len(df) / 10
        ax.axhline(linea_esperada, color=COLORES["rojo"],
                   linestyle="--", linewidth=1.2, alpha=0.7, label=f"Esperado ({linea_esperada:.0f})")
        ax.legend(fontsize=8)
        for bar, f in zip(bars, frecuencias):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(f), ha="center", va="bottom", fontsize=8,
                    color=COLORES["azul_oscuro"], fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight",
                facecolor=COLORES["fondo"])
    plt.close()
    print(f"  ✅ Guardado: {ruta_salida}")

def grafico_tendencia_anual(df, ruta_salida):
    df2 = df.copy()
    df2["anio"] = df2["Fecha"].dt.year
    df2["numero_int"] = df2["Numero"].astype(int)

    anios = sorted(df2["anio"].unique())
    promedios = [df2[df2["anio"]==a]["numero_int"].mean() for a in anios]
    stds      = [df2[df2["anio"]==a]["numero_int"].std() for a in anios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORES["fondo"])

    # — Promedio por año —
    ax1.set_facecolor(COLORES["fondo"])
    ax1.plot(anios, promedios, color=COLORES["azul_oscuro"],
             linewidth=2.5, marker="o", markersize=7, zorder=3)
    ax1.fill_between(anios,
                     [p - s for p, s in zip(promedios, stds)],
                     [p + s for p, s in zip(promedios, stds)],
                     alpha=0.15, color=COLORES["azul_medio"])
    ax1.axhline(4999.5, color=COLORES["rojo"], linestyle="--",
                linewidth=1, alpha=0.6, label="Media teórica (5000)")
    ax1.set_title("Promedio del número ganador por año",
                  fontsize=12, fontweight="bold", color=COLORES["azul_oscuro"])
    ax1.set_xlabel("Año", fontsize=10, color=COLORES["gris"])
    ax1.set_ylabel("Número promedio", fontsize=10, color=COLORES["gris"])
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)
    for x, y in zip(anios, promedios):
        ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8,
                     color=COLORES["azul_oscuro"])

    # — Distribución general del número completo —
    ax2.set_facecolor(COLORES["fondo"])
    ax2.hist(df2["numero_int"], bins=40, color=COLORES["azul_medio"],
             edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax2.set_title("Distribución de números ganadores (0000-9999)",
                  fontsize=12, fontweight="bold", color=COLORES["azul_oscuro"])
    ax2.set_xlabel("Número", fontsize=10, color=COLORES["gris"])
    ax2.set_ylabel("Frecuencia", fontsize=10, color=COLORES["gris"])
    ax2.grid(alpha=0.3, zorder=0)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("ANÁLISIS TEMPORAL — LOTERÍA DE SANTANDER",
                 fontsize=14, fontweight="bold",
                 color=COLORES["azul_oscuro"], y=1.02)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight",
                facecolor=COLORES["fondo"])
    plt.close()
    print(f"  ✅ Guardado: {ruta_salida}")



def grafico_prediccion(probabilidades, fecha_sorteo, cv_scores, ruta_salida):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORES["fondo"])

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.35,
                           height_ratios=[0.6, 2.5, 1.2])

    fecha_str = fecha_sorteo.strftime("%d de %B de %Y") if hasattr(fecha_sorteo, "strftime") \
                else str(fecha_sorteo)

    # — Título —
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.7,
                  f"🎯  PRONÓSTICO LOTERÍA SANTANDER",
                  ha="center", va="center", fontsize=18,
                  fontweight="bold", color=COLORES["azul_oscuro"],
                  transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15,
                  f"Sorteo estimado: {fecha_str}  |  "
                  f"Modelos: Random Forest + XGBoost (Ensemble)",
                  ha="center", va="center", fontsize=10,
                  color=COLORES["gris"], transform=ax_title.transAxes)

    pos_names   = ["1ª Cifra", "2ª Cifra", "3ª Cifra", "4ª Cifra"]
    digitos     = list(range(10))
    num_predicho = ""

    for idx in range(4):
        pos  = idx + 1
        probs = probabilidades[pos]
        best = int(np.argmax(probs))
        num_predicho += str(best)

        ax = fig.add_subplot(gs[1, idx])
        ax.set_facecolor(COLORES["fondo"])

        colores_bar = []
        for d in digitos:
            if d == best:
                colores_bar.append(COLORES["azul_oscuro"])
            elif probs[d] >= np.percentile(probs, 70):
                colores_bar.append(COLORES["azul_medio"])
            else:
                colores_bar.append(COLORES["azul_claro"])

        bars = ax.bar(digitos, probs * 100, color=colores_bar,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.axhline(10, color=COLORES["rojo"], linestyle="--",
                   linewidth=1, alpha=0.6, label="Azar puro (10%)")
        ax.set_title(f"{pos_names[idx]}\n→ Predicción: {best}  ({probs[best]*100:.1f}%)",
                     fontsize=11, fontweight="bold", color=COLORES["azul_oscuro"],
                     pad=6)
        ax.set_xlabel("Dígito", fontsize=9, color=COLORES["gris"])
        ax.set_ylabel("Probabilidad (%)", fontsize=9, color=COLORES["gris"])
        ax.set_xticks(digitos)
        max_prob = np.nanmax(probs)

        if np.isnan(max_prob) or max_prob == 0:
            max_prob = 0.1

        ax.set_ylim(0, max_prob * 115)

        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7)
        for bar, p in zip(bars, probs):
            if p * 100 >= 5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{p*100:.1f}%", ha="center", va="bottom",
                        fontsize=7, color=COLORES["azul_oscuro"],
                        fontweight="bold")

    # — Número predicho y métricas —
    ax_result = fig.add_subplot(gs[2, :])
    ax_result.axis("off")

    # Caja del número
    rect = FancyBboxPatch((0.2, 0.05), 0.6, 0.85,
                          boxstyle="round,pad=0.02",
                          facecolor="#EBF3FB", edgecolor=COLORES["azul_oscuro"],
                          linewidth=2, transform=ax_result.transAxes, zorder=2)
    ax_result.add_patch(rect)

    ax_result.text(0.5, 0.72,
                   f"NÚMERO PREDICHO: {num_predicho}",
                   ha="center", va="center", fontsize=20,
                   fontweight="bold", color=COLORES["azul_oscuro"],
                   transform=ax_result.transAxes, zorder=3)

    prob_total = np.prod([probabilidades[p+1][int(num_predicho[p])] for p in range(4)])
    cv_medio   = np.mean([cv_scores[p]["cv_xgb"] for p in range(1, 5)])

    ax_result.text(0.5, 0.38,
                   f"Probabilidad combinada de este número exacto: {prob_total*100:.4f}%   |   "
                   f"Precisión promedio del modelo (CV): {cv_medio:.1%}",
                   ha="center", va="center", fontsize=10,
                   color=COLORES["gris"], transform=ax_result.transAxes, zorder=3)

    ax_result.text(0.5, 0.10,
                   "⚠️  Este pronóstico es estadístico/exploratorio. La lotería es aleatoria "
                   "y ningún modelo garantiza resultados.",
                   ha="center", va="center", fontsize=8,
                   color=COLORES["rojo"], style="italic",
                   transform=ax_result.transAxes, zorder=3)

    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight",
                facecolor=COLORES["fondo"])
    plt.close()
    print(f"  ✅ Guardado: {ruta_salida}")
    return num_predicho, prob_total

def grafico_heatmap_correlacion(df, ruta_salida):
    """Heatmap: ¿el dígito de esta semana depende del de la anterior?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor(COLORES["fondo"])
    fig.suptitle("¿HAY CORRELACIÓN ENTRE SORTEOS CONSECUTIVOS?\n"
                 "Dígito actual vs. dígito del sorteo anterior (por posición)",
                 fontsize=13, fontweight="bold", color=COLORES["azul_oscuro"])

    pos_names = ["1ª Cifra", "2ª Cifra", "3ª Cifra", "4ª Cifra"]
    for idx, (ax, pname) in enumerate(zip(axes.flat, pos_names)):
        pos  = idx + 1
        col  = f"D{pos}"
        prev = df[col].shift(1).dropna().astype(int)
        curr = df[col].iloc[1:].astype(int)

        matriz = np.zeros((10, 10))
        for p, c in zip(prev, curr):
            matriz[p][c] += 1

        im = ax.imshow(matriz, cmap="Blues", aspect="auto")
        ax.set_facecolor(COLORES["fondo"])
        ax.set_title(f"{pname}", fontsize=11, fontweight="bold",
                     color=COLORES["azul_oscuro"])
        ax.set_xlabel("Dígito actual", fontsize=9)
        ax.set_ylabel("Dígito anterior", fontsize=9)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Anotaciones
        for i in range(10):
            for j in range(10):
                v = int(matriz[i][j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=6, color="white" if v > matriz.max()*0.6 else "black")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight",
                facecolor=COLORES["fondo"])
    plt.close()
    print(f"  ✅ Guardado: {ruta_salida}")

# ═══════════════════════════════════════════════════════════
#  6. REPORTE DE TEXTO
# ═══════════════════════════════════════════════════════════

def imprimir_reporte(probabilidades, num_predicho, prob_total, fecha_sorteo, cv_scores, df):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  PRONÓSTICO LOTERÍA DE SANTANDER")
    fecha_str = fecha_sorteo.strftime("%d/%m/%Y") if hasattr(fecha_sorteo, "strftime") \
                else str(fecha_sorteo)
    print(f"  Sorteo estimado: {fecha_str}")
    print(sep)

    pos_names = ["1ª Cifra", "2ª Cifra", "3ª Cifra", "4ª Cifra"]
    for pos in range(1, 5):
        probs   = probabilidades[pos]
        top3    = sorted(range(10), key=lambda d: probs[d], reverse=True)[:3]
        print(f"\n  {pos_names[pos-1]}:")
        for rank, d in enumerate(top3, 1):
            bar = "█" * int(probs[d] * 100)
            print(f"    #{rank}  Dígito {d}  {probs[d]*100:5.2f}%  {bar}")

    print(f"\n{'-'*60}")
    print(f"  NÚMERO PREDICHO:  ➤  {num_predicho}  ◄")
    print(f"  Prob. combinada del número exacto: {prob_total*100:.6f}%")
    print(f"  (Prob. teórica azar puro:          0.010000%)")
    print(f"\n  Precisión del modelo por posición (Cross-Validation):")
    for pos in range(1, 5):
        print(f"    {pos_names[pos-1]}: RF={cv_scores[pos]['cv_rf']:.1%}  "
              f"XGB={cv_scores[pos]['cv_xgb']:.1%}")

    # Top 5 números más probables
    print(f"\n  TOP 5 NÚMEROS MÁS PROBABLES:")
    combos = []
    for n0 in range(10):
        for n1 in range(10):
            for n2 in range(10):
                for n3 in range(10):
                    p = (probabilidades[1][n0] * probabilidades[2][n1] *
                         probabilidades[3][n2] * probabilidades[4][n3])
                    combos.append((f"{n0}{n1}{n2}{n3}", p))
    combos.sort(key=lambda x: x[1], reverse=True)
    for i, (num, p) in enumerate(combos[:5], 1):
        print(f"    #{i}  {num}  →  {p*100:.6f}%")

    print(f"\n  ⚠️  Recuerda: la lotería es aleatoria.")
    print(f"     Este análisis es exploratorio/académico.")
    print(sep)

# ═══════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════

def main():
    if os.path.exists(HIST_FILE):
        historial = pd.read_excel(HIST_FILE)
    else:
        historial = pd.DataFrame()

    # 🔥 asegurar columnas SIEMPRE
    for col in ["fecha", "predicho", "real", "aciertos", "detalle"]:
        if col not in historial.columns:
            historial[col] = ""

    print("\n" + "="*60)
    print("  PREDICTOR LOTERÍA SANTANDER — Iniciando...")
    print("="*60)

    # — Cargar datos —
    if not os.path.exists(RUTA_EXCEL):
        print(f"[ERROR] No se encontró el Excel en:\n  {RUTA_EXCEL}")
        print("Coloca el archivo en la misma carpeta que este script.")
        sys.exit(1)

    print(f"\n📂 Cargando datos de: {RUTA_EXCEL}")
    df = cargar_datos(RUTA_EXCEL)
    print(f"   {len(df)} sorteos cargados  ({df['Fecha'].min().date()} → {df['Fecha'].max().date()})")

    # — Features y modelos —
    print("\n🔧 Generando características...")
    feat_df = crear_features(df, ventana=10)
    print(f"   {len(feat_df)} muestras de entrenamiento")

    print("\n🤖 Entrenando modelos (RF + XGBoost)...")
    modelos = entrenar_modelos(feat_df)

    # — Predicción próximo viernes —
    print("\n🔮 Calculando predicción...")
    probs, fecha_sorteo = predecir_para_fecha(df, modelos, historial=historial)

    # — Gráficos —
    print("\n📊 Generando gráficos...")
    g1 = os.path.join(CARPETA_SALIDA, "grafico_frecuencias.png")
    g2 = os.path.join(CARPETA_SALIDA, "grafico_tendencia.png")
    g3 = os.path.join(CARPETA_SALIDA, "grafico_prediccion.png")
    g4 = os.path.join(CARPETA_SALIDA, "grafico_correlacion.png")

    grafico_frecuencias_historicas(df, g1)
    grafico_tendencia_anual(df, g2)
    num_pred, prob_total = grafico_prediccion(probs, fecha_sorteo, modelos, g3)
    grafico_heatmap_correlacion(df, g4)

    # — Reporte —
    imprimir_reporte(probs, num_pred, prob_total, fecha_sorteo, modelos, df)

    os.makedirs(os.path.dirname(HIST_FILE), exist_ok=True)

    # ── Guardar predicción en historial ──

    predicho = num_pred
    fecha = fecha_sorteo.strftime("%Y-%m-%d")

    nuevo = pd.DataFrame({
        "fecha":[fecha],
        "predicho":[predicho],
        "real":[""],
        "aciertos":[0],
        "detalle":[""]
    })

    if os.path.exists(HIST_FILE):

        historial = pd.read_excel(HIST_FILE)

        # 🔥 FORZAR TIPOS CORRECTOS
        historial["predicho"] = historial["predicho"].astype(str)
        historial["real"] = historial["real"].astype(str)
        historial["detalle"] = historial["detalle"].astype(str)

        # asegurar columnas DESPUÉS de cargar
        for col in ["fecha", "predicho", "real", "detalle"]:
            if col not in historial.columns:
                historial[col] = ""

        if "aciertos" not in historial.columns:
            historial["aciertos"] = 0

        historial = actualizar_resultados(historial, df)
        print("\n📊 RESULTADOS RECIENTES:")
        print("-"*50)

        ultimos = historial.tail(5)

        for _, row in ultimos.iterrows():
            print(f"Fecha: {row['fecha']}")
            print(f"Predicho: {row['predicho']}  |  Real: {row['real']}")
            detalle = row["detalle"] if "detalle" in row else ""
            print(f"Aciertos: {row['aciertos']}  |  Detalle: {detalle}")    
            print("-"*50)

        # evitar duplicado del mismo día
        if fecha not in historial["fecha"].astype(str).values:
            historial = pd.concat([historial, nuevo], ignore_index=True)

    else:

        historial = nuevo

    historial.to_excel(HIST_FILE, index=False)

    if os.path.exists(HIST_FILE):
        historial = pd.read_excel(HIST_FILE)
    else:
        historial = pd.DataFrame(columns=["fecha","predicho","real","aciertos","detalle"])

    print(f"\n📝 Predicción guardada en historial: {predicho}")

    print(f"\n✅ Todo listo. Archivos generados en:\n   {CARPETA_SALIDA}")
    print("   • grafico_frecuencias.png")
    print("   • grafico_tendencia.png")
    print("   • grafico_prediccion.png   ← el más importante")
    print("   • grafico_correlacion.png")

def actualizar_resultados(historial, df_real):

    for i, row in historial.iterrows():

        # Solo actualizar si aún no tiene resultado
        if row["real"] == "" or pd.isna(row["real"]):

            fecha = pd.to_datetime(row["fecha"]).date()

            match = df_real[
                df_real["Fecha"].dt.date == fecha
            ]

            if not match.empty:

                real = str(match.iloc[0]["Numero"]).zfill(4)

                aciertos, detalle = analizar_resultado(str(row["predicho"]), real)

                historial.at[i, "real"] = real
                historial.at[i, "aciertos"] = aciertos
                historial.at[i, "detalle"] = detalle

                print(f"✔ Actualizado {fecha} → Real: {real} | Aciertos: {aciertos}")

            else:
                print(f"⏳ Aún no hay resultado para {fecha}")

    return historial


if __name__ == "__main__":
    main()
