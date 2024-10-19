import os
import sys
from sklearn.tree import DecisionTreeClassifier
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
import yaml
import mlflow

def trainTree(X, Y, **kwargs):
        """
        Función para entrenar un modelo DecisionTreeClassifier utilizando búsqueda de hiperparámetros
        con GridSearchCV.

        Parámetros:
        -----------
        X : numpy array o pandas DataFrame
            Matriz de características (features) de los datos de entrenamiento.

        Y : numpy array o pandas Series
            Etiquetas (labels) correspondientes a los datos de entrenamiento.

        criterion : list
            Funcion para medir la calidad de un split.
        max_depth : list
            Profundidad maxima de un arbol
        min_samples_split : int
            Minimo de muestras requeridas para dividir un nodo
        Retorno:
        --------
        best_decision_tree_model : DecisionTreeClassifier
            El mejor modelo DecisionTreeClassifier encontrado durante la búsqueda de hiperparámetros utilizando GridSearchCV.
        """

        parameters = {
            'criterion': kwargs["criterion"],
            'max_depth': kwargs["max_depth"],
            'min_samples_split': kwargs["min_samples_split"],
        }

        # Realizar búsqueda de hiperparámetros usando GridSearchCV
        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=kwargs["seed"]),
                                param_grid=parameters,
                                scoring="f1_weighted",
                                verbose=0,
                                n_jobs=-1)   

        # Ajustar el modelo con los datos de entrenamiento
        grid_search.fit(X, Y)

        # Obtener los mejores hiperparámetros encontrados
        best_decision_tree_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        return best_decision_tree_model, best_params

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
    params = yaml.safe_load(open("params.yaml"))["train_decision_tree"]
    version_label = yaml.safe_load(open("params.yaml"))["version_label"]
    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    # Inicializar la clase ClassifierModel con la versión y semilla
    classifier_model = trainTree(X_train, y_train, **params)

    # Inicializar MLFlow
    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.set_tracking_uri(mlflow_params['tracking_uri']) 

    # Asignar el nombre del run usando el formato definido en params.yaml
    run_name = f"Train Decision tree"

    # Entrenar y guardar el modelo DecisionTree usando los parámetros de `params.yaml`
    with mlflow.start_run(run_name=run_name) as run:
        decision_tree_model, best_params = trainTree(X_train, y_train, **params     
        )

        # Guardar los mejores hiperparámetros en MLflow
        mlflow.log_params(best_params)

        # Verificar si el modelo tiene un atributo best_params_
        #if hasattr(decision_tree_model, "best_params_"):
            # Guardar solo los mejores hiperparámetros en MLflow
        #    mlflow.log_params(decision_tree_model.best_params_)
        #else:
            # Si no hay una búsqueda de hiperparámetros, guardar manualmente los valores actuales
        #    mlflow.log_param("n_neighbors", params["n_neighbors"])
        #    mlflow.log_param("weights", params["weights"])
        #    mlflow.log_param("distance_metric", params["p"])

        # Guardar el modelo en MLFlow
        mlflow.sklearn.log_model(decision_tree_model, f"DT_model")
    
        # Guardar el run_id para compartirlo con evaluate.py
        run_id = run.info.run_id
        with open(os.path.join(out_path, "mlflow_run_id.txt"), "w") as f:
            f.write(run_id)

    # Guardar el modelo en el directorio de salida
    model_output_path = os.path.join(out_path, "decision_tree_model.pkl")
    joblib.dump(decision_tree_model, model_output_path)
    sys.stderr.write(f"Modelo Decision Tree guardado en {model_output_path}\n")

if __name__ == "__main__":
    main()
