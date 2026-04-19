import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import HIST_FILE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def contar_aciertos(predicho, real):
    return sum(p == r for p, r in zip(str(predicho), str(real)))


def grafico_aciertos(df):

    plt.plot(df["aciertos"])

    plt.title("Aciertos del modelo en el tiempo")
    plt.xlabel("Sorteo")
    plt.ylabel("Aciertos")

    plt.show()


def evaluar():

    if not os.path.exists(HIST_FILE):
        print("No hay historial aún.")
        return

    df = pd.read_excel(HIST_FILE)

    total = len(df)
    exactos = sum(df["aciertos"] == 4)

    print("\n========== EVALUACIÓN DEL MODELO ==========")
    print("Total predicciones:", total)
    print("Exactos:", exactos)

    grafico_aciertos(df)


if __name__ == "__main__":
    evaluar()