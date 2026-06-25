import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENTO = "classificacao-vinho"
NOME_MODELO = "classificador-vinho"


def encontrar_melhor_run() -> tuple[str, float]:
    client = MlflowClient()
    experimento = client.get_experiment_by_name(EXPERIMENTO)

    if experimento is None:
        raise RuntimeError(f"Experimento '{EXPERIMENTO}' nao encontrado. Execute treinar.py primeiro.")

    runs = client.search_runs(
        experiment_ids=[experimento.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("Nenhum run encontrado no experimento.")

    melhor = runs[0]
    return melhor.info.run_id, melhor.data.metrics["accuracy"]


def registrar_melhor_modelo() -> None:
    client = MlflowClient()
    run_id, accuracy = encontrar_melhor_run()

    print(f"Melhor run: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")

    uri_modelo = f"runs:/{run_id}/modelo"
    resultado = mlflow.register_model(model_uri=uri_modelo, name=NOME_MODELO)

    versao = resultado.version
    print(f"Modelo '{NOME_MODELO}' registrado como versao {versao}")

    client.set_registered_model_alias(
        name=NOME_MODELO,
        alias="champion",
        version=versao,
    )
    print(f"Alias 'champion' atribuido a versao {versao}")
    print(f"\nPara carregar: mlflow.sklearn.load_model('models:/{NOME_MODELO}@champion')")


if __name__ == "__main__":
    registrar_melhor_modelo()
