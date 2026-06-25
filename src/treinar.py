import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preparar_dados import preparar

EXPERIMENTO = "classificacao-vinho"

MODELOS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "ridge_classifier": RidgeClassifier(alpha=1.0),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "f1_macro": f1_score(y_real, y_pred, average="macro"),
        "precision_macro": precision_score(y_real, y_pred, average="macro"),
        "recall_macro": recall_score(y_real, y_pred, average="macro"),
    }


def treinar_modelo(nome: str, modelo, X_treino, X_teste, y_treino, y_teste) -> str:
    with mlflow.start_run(run_name=nome) as run:
        mlflow.log_params(modelo.get_params())

        cv_scores = cross_val_score(modelo, X_treino, y_treino, cv=5, scoring="accuracy")
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_teste)

        metricas = calcular_metricas(y_teste, y_pred)
        mlflow.log_metrics(metricas)

        mlflow.sklearn.log_model(modelo, name="modelo", registered_model_name=None)

        print(
            f"  {nome:25s} | acc: {metricas['accuracy']:.4f} "
            f"| f1: {metricas['f1_macro']:.4f} "
            f"| cv: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

        return run.info.run_id


def treinar_todos() -> dict:
    mlflow.set_experiment(EXPERIMENTO)

    X_treino, X_teste, y_treino, y_teste, scaler, colunas = preparar()

    print(f"\nExperimento: {EXPERIMENTO}")
    print(f"Treino: {X_treino.shape} | Teste: {X_teste.shape}\n")
    print(f"{'Modelo':<25} | {'Accuracy':>8} | {'F1 Macro':>8} | {'CV Mean':>8} | {'CV Std':>8}")
    print("-" * 70)

    run_ids = {}
    for nome, modelo in MODELOS.items():
        run_id = treinar_modelo(nome, modelo, X_treino, X_teste, y_treino, y_teste)
        run_ids[nome] = run_id

    print(f"\nTodos os runs registrados no MLflow.")
    print("Para visualizar: mlflow ui")
    return run_ids


if __name__ == "__main__":
    treinar_todos()
