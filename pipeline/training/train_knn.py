import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../model/classifiermodel.py
from model.classifiermodel import ClassifierModel

import joblib
import pandas as pd
import yaml

import mlflow


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py preprocessed-dir-path model-dir-path\n")
        sys.exit(1)

    in_path = sys.argv[1]  # Ruta de los datos preprocesados
    out_path = sys.argv[2]  # Ruta donde se guardará el modelo entrenado

    # Crear el directorio para guardar el modelo si no existe
    os.makedirs(out_path, exist_ok=True)

    # Cargar los datos preprocesados (train)
    train_data = pd.read_csv(os.path.join(in_path, "train.csv"))

    # Separar las características (X) y el target (y) de los datos preprocesados
    X_train = train_data.drop(columns=['0'])
    y_train = train_data['0']

    # Cargar los parámetros del archivo params.yaml
    params = yaml.safe_load(open("params.yaml"))["train_knn"]
    version_label = yaml.safe_load(open("params.yaml"))["version_label"]
    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    # Inicializar la clase ClassifierModel con la versión y semilla
    classifier_model = ClassifierModel(versionLabel=version_label, randomSeed=params["seed"])

    # Inicializar MLFlow
    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.set_tracking_uri(mlflow_params['tracking_uri']) 

    # Asignar el nombre del run usando el formato definido en params.yaml
    run_name = mlflow_params["run_name"].format(model="KNN")

    # Entrenar y guardar el modelo KNN usando los parámetros de `params.yaml`
    with mlflow.start_run(run_name=run_name) as run:
        knn_model, best_params = classifier_model.trainKNN(
            X_train, 
            y_train, 
            n_neighbors=params["n_neighbors"],   
            weights=params["weights"],           
            Distance=params["p"]           
        )

        # Guardar los mejores hiperparámetros en MLflow
        mlflow.log_params(best_params)

        # Verificar si el modelo tiene un atributo best_params_
        #if hasattr(knn_model, "best_params_"):
            # Guardar solo los mejores hiperparámetros en MLflow
        #    mlflow.log_params(knn_model.best_params_)
        #else:
            # Si no hay una búsqueda de hiperparámetros, guardar manualmente los valores actuales
        #    mlflow.log_param("n_neighbors", params["n_neighbors"])
        #    mlflow.log_param("weights", params["weights"])
        #    mlflow.log_param("distance_metric", params["p"])

        # Guardar el modelo en MLFlow
        mlflow.sklearn.log_model(knn_model, f"KNN_model")
    
        # Guardar el run_id para compartirlo con evaluate.py
        run_id = run.info.run_id
        with open(os.path.join(out_path, "mlflow_run_id_knn_model.txt"), "w") as f:
            f.write(run_id)

    # Guardar el modelo en el directorio de salida
    model_output_path = os.path.join(out_path, "knn_model.pkl")
    joblib.dump(knn_model, model_output_path)
    sys.stderr.write(f"Modelo KNN guardado en {model_output_path}\n")

if __name__ == "__main__":
    main()
