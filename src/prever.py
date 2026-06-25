import mlflow.sklearn
import numpy as np
import pandas as pd

NOME_MODELO = "classificador-vinho"
ALIAS = "champion"

CLASSES = {0: "Classe 1 (premium)", 1: "Classe 2 (medio)", 2: "Classe 3 (entrada)"}


def carregar_modelo():
    uri = f"models:/{NOME_MODELO}@{ALIAS}"
    print(f"Carregando modelo: {uri}")
    return mlflow.sklearn.load_model(uri)


def prever(amostra: dict) -> str:
    modelo = carregar_modelo()
    X = np.array([[v for v in amostra.values()]])
    predicao = modelo.predict(X)[0]
    probabilidades = modelo.predict_proba(X)[0] if hasattr(modelo, "predict_proba") else None

    print(f"\nAmostra: {amostra}")
    print(f"Predicao: {predicao} - {CLASSES.get(predicao, 'desconhecida')}")
    if probabilidades is not None:
        for i, prob in enumerate(probabilidades):
            print(f"  {CLASSES[i]}: {prob:.2%}")

    return CLASSES.get(predicao, "desconhecida")


if __name__ == "__main__":
    # amostra de exemplo com valores tipicos do dataset Wine
    amostra_exemplo = {
        "alcohol": 13.2,
        "malic_acid": 1.78,
        "ash": 2.14,
        "alcalinity_of_ash": 11.2,
        "magnesium": 100.0,
        "total_phenols": 2.65,
        "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26,
        "proanthocyanins": 1.28,
        "color_intensity": 4.38,
        "hue": 1.05,
        "od280/od315_of_diluted_wines": 3.40,
        "proline": 1050.0,
        "alcool_acido": 13.2 / (1.78 + 1e-6),
        "cor_por_alcool": 4.38 / (13.2 + 1e-6),
        "indice_maturidade": 2.76 * 2.65,
    }
    prever(amostra_exemplo)
