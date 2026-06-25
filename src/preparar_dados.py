import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def carregar_dataset() -> pd.DataFrame:
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["qualidade"] = wine.target
    return df


def engenharia_de_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # razao alcool/acido - indica perfil do vinho
    df["alcool_acido"] = df["alcohol"] / (df["malic_acid"] + 1e-6)
    # intensidade de cor normalizada pelo teor de alcool
    df["cor_por_alcool"] = df["color_intensity"] / (df["alcohol"] + 1e-6)
    # indice de maturidade - flavonoids altos indicam uva mais madura
    df["indice_maturidade"] = df["flavanoids"] * df["total_phenols"]
    return df


def preparar(test_size: float = 0.2, random_state: int = 42) -> tuple:
    df = carregar_dataset()
    df = engenharia_de_features(df)

    X = df.drop(columns=["qualidade"])
    y = df["qualidade"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    return X_treino_scaled, X_teste_scaled, y_treino, y_teste, scaler, X.columns.tolist()


def salvar_csv(caminho: str = "dados/wine.csv") -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    df = carregar_dataset()
    df = engenharia_de_features(df)
    df.to_csv(caminho, index=False)
    print(f"Dataset salvo em {caminho} - {len(df)} amostras, {df.shape[1]} colunas")


if __name__ == "__main__":
    salvar_csv()
    X_treino, X_teste, y_treino, y_teste, scaler, colunas = preparar()
    print(f"Treino: {X_treino.shape} | Teste: {X_teste.shape}")
    print(f"Classes: {y_treino.unique()}")
