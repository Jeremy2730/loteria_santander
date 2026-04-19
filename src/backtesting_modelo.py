import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import EXCEL_FILE as DATA_FILE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def contar_aciertos(predicho, real):
    return sum(p == r for p, r in zip(str(predicho), str(real)))


def backtesting():

    df = pd.read_excel(DATA_FILE)

    numeros = df["numero"].astype(str).str.zfill(4)

    resultados = []

    ventana = 200

    for i in range(ventana, len(numeros)-1):

        train = numeros[:i]

        X = []
        y = []

        for j in range(len(train)-1):
            X.append([int(d) for d in train[j]])
            y.append([int(d) for d in train[j+1]])

        X = np.array(X)
        y = np.array(y)

        modelos = []

        pred = ""

        for pos in range(4):

            model = RandomForestClassifier(n_estimators=200)

            model.fit(X, y[:,pos])

            p = model.predict([X[-1]])

            pred += str(p[0])

        real = numeros[i]

        aciertos = contar_aciertos(pred, real)

        resultados.append(aciertos)

        print(f"Predicho {pred}  Real {real}  Aciertos {aciertos}")

    print("\n===========================")
    print("RESULTADOS BACKTESTING")
    print("===========================")

    resultados = np.array(resultados)

    print("Promedio aciertos:", resultados.mean())
    print("Exactos:", sum(resultados==4))


if __name__ == "__main__":
    backtesting()