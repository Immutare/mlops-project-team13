import os
import sys

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_logistic(X, y, **kwargs):
    parameters = {
        "C": kwargs["C"],
        "max_iter": kwargs["max_iter"],
        "solver": kwargs["solver"],
    }

    # Realizar búsqueda de hiperparámetros usando GridSearchCV
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=kwargs["seed"]),
        param_grid=parameters,
        scoring="f1_weighted",
        verbose=0,
        n_jobs=-1,
    )

    # Ajustar el modelo con los datos de entrenamiento
    grid_search.fit(X, y)

    # Obtener el mejor modelo encontrado
    best_logistic_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_logistic_model, best_params


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            "\tpython train_logistic.py preprocessed-dir-path model-dir-path\n"
        )
        sys.exit(1)

    in_path = sys.argv[1]  # Ruta de los datos preprocesados
    out_path = sys.argv[2]  # Ruta donde se guardará el modelo entrenado

    # Crear el directorio para guardar el modelo si no existe
    os.makedirs(out_path, exist_ok=True)

    # Cargar los datos preprocesados (train)
    train_data = pd.read_csv(os.path.join(in_path, "train.csv"))

    # Separar las características (X) y el target (y) de los datos
    # preprocesados
    X_train = train_data.drop(columns=["RiskLevel"])
    y_train = train_data["RiskLevel"]

    # Cargar los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml"))["train_logistic"]
    # version_label = yaml.safe_load(open("params.yaml"))["version_label"]
    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    # Inicializar MLFlow
    mlflow.set_experiment(mlflow_params["experiment_name"])
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    # Asignar el nombre del run usando el formato definido en params.yaml
    run_name = "Train Logistic Regression"

    # Entrenar y guardar el modelo Logistic Regression usando los parámetros
    # de `params.yaml`
    with mlflow.start_run(run_name=run_name) as run:
        logistic_model, best_params = train_logistic(
            X_train, y_train, **params
        )

        # Guardar los mejores hiperparámetros en MLflow
        mlflow.log_params(best_params)

        # Guardar el modelo en MLFlow
        mlflow.sklearn.log_model(logistic_model, "logistic_model")

        # Guardar el run_id para compartirlo con evaluate.py
        run_id = run.info.run_id
        with open(
            os.path.join(out_path, "mlflow_run_id_logistic_model.txt"), "w"
        ) as f:
            f.write(run_id)

    # Guardar el modelo en el directorio de salida
    model_output_path = os.path.join(out_path, "logistic_model.pkl")
    joblib.dump(logistic_model, model_output_path)
    sys.stderr.write(
        f"Modelo Logistic Regression guardado en {model_output_path}\n"
    )


if __name__ == "__main__":
    main()
