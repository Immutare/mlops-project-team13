import os
import sys

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import GridSearchCV


def train_xgboost(X, y, **kwargs):

    parameters = {
        "max_depth": kwargs["max_depth"],
        "learning_rate": kwargs["learning_rate"],
        "subsample": kwargs["subsample"],
    }

    # Create the XGBoost model object
    xgb_model = xgb.XGBClassifier()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        xgb_model, parameters, cv=5, scoring=kwargs["scoring"]
    )

    # Ajustar el modelo con los datos de entrenamiento
    grid_search.fit(X.values, y)

    # Obtener el mejor modelo encontrado
    best_xgboost_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_xgboost_model, best_params


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            "\tpython train_xgboost.py preprocessed-dir-path model-dir-path\n"
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
    X_train = train_data.drop(columns=["0"])  # Cambia 'RiskLevel' a '0'
    y_train = train_data["0"]  # Cambia 'RiskLevel' a '0'

    # Cargar los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml"))["train_xgboost"]
    # version_label = yaml.safe_load(open("params.yaml"))["version_label"]
    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    # Inicializar MLFlow
    mlflow.set_experiment(mlflow_params["experiment_name"])
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    # Asignar el nombre del run usando el formato definido en params.yaml
    run_name = "Train XGBoost"

    # Entrenar y guardar el modelo XGBoost usando los parámetros de
    # `params.yaml`
    with mlflow.start_run(run_name=run_name) as run:
        xgboost_model, best_params = train_xgboost(X_train, y_train, **params)

        # Guardar los mejores hiperparámetros en MLflow
        mlflow.log_params(best_params)

        # Guardar el modelo en MLFlow
        mlflow.sklearn.log_model(xgboost_model, "XGBoost")

        # Guardar el run_id para compartirlo con evaluate.py
        run_id = run.info.run_id
        with open(
            os.path.join(out_path, "mlflow_run_id_xgboost_model.txt"), "w"
        ) as f:
            f.write(run_id)

    # Guardar el modelo en el directorio de salida
    model_output_path = os.path.join(out_path, "xgboost_model.pkl")
    joblib.dump(xgboost_model, model_output_path)
    sys.stderr.write(f"Modelo 'XGBoost' guardado en {model_output_path}\n")


if __name__ == "__main__":
    main()
