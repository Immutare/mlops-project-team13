import os
import sys
from sklearn.svm import SVC
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
import yaml
import mlflow

def trainSVM(X, Y, **kwargs):
    """
    Función para entrenar un modelo SVM utilizando búsqueda de hiperparámetros
    con GridSearchCV.

    Parámetros:
    -----------
    X : numpy array o pandas DataFrame
        Matriz de características (features) de los datos de entrenamiento.

    Y : numpy array o pandas Series
        Etiquetas (labels) correspondientes a los datos de entrenamiento.

    C : list
        Valores de regularización del modelo SVM.
    kernel : list
        Tipo de kernel a usar (e.g., 'linear', 'rbf', etc.)
    gamma : list
        Parámetro de kernel para modelos no lineales.
    
    Retorno:
    --------
    best_svm_model : SVC
        El mejor modelo SVM encontrado durante la búsqueda de hiperparámetros utilizando GridSearchCV.
    """

    parameters = {
        'C': kwargs["C"],
        'kernel': kwargs["kernel"],
        'gamma': kwargs["gamma"],
    }

    # Realizar búsqueda de hiperparámetros usando GridSearchCV
    grid_search = GridSearchCV(estimator=SVC(random_state=kwargs["seed"]),
                               param_grid=parameters,
                               scoring="f1_weighted",
                               verbose=0,
                               n_jobs=-1)

    # Ajustar el modelo con los datos de entrenamiento
    grid_search.fit(X, Y)

    # Obtener los mejores hiperparámetros encontrados
    best_svm_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_svm_model, best_params

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
    params = yaml.safe_load(open("params.yaml"))["train_svm"]
    version_label = yaml.safe_load(open("params.yaml"))["version_label"]
    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    # Inicializar la clase ClassifierModel con la versión y semilla
    classifier_model = trainSVM(X_train, y_train, **params)

    # Inicializar MLFlow
    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.set_tracking_uri(mlflow_params['tracking_uri']) 

    # Asignar el nombre del run usando el formato definido en params.yaml
    run_name = mlflow_params["run_name"].format(model="SVM")

    # Entrenar y guardar el modelo SVM usando los parámetros de `params.yaml`
    with mlflow.start_run(run_name=run_name) as run:
        svm_model, best_params = trainSVM(X_train, y_train, **params)

        # Guardar los mejores hiperparámetros en MLflow
        mlflow.log_params(best_params)

        # Guardar el modelo en MLFlow
        mlflow.sklearn.log_model(svm_model, f"SVM_model")
    
        # Guardar el run_id para compartirlo con evaluate.py
        run_id = run.info.run_id
        with open(os.path.join(out_path, "mlflow_run_id_svm_model.txt"), "w") as f:
            f.write(run_id)

    # Guardar el modelo en el directorio de salida
    model_output_path = os.path.join(out_path, "svm_model.pkl")
    joblib.dump(svm_model, model_output_path)
    sys.stderr.write(f"Modelo SVM guardado en {model_output_path}\n")

if __name__ == "__main__":
    main()
