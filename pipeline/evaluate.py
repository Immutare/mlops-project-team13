import os
import sys
import joblib
import pandas as pd
import yaml
import mlflow
from model.classifiermodel import ClassifierModel

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py preprocessed-dir-path model-dir-path\n")
        sys.exit(1)

    in_path = sys.argv[1]  # Ruta de los datos preprocesados (test)
    model_path = sys.argv[2]  # Ruta donde se guardó el modelo entrenado

    # Cargar los datos preprocesados (test)
    test_data = pd.read_csv(os.path.join(in_path, "test.csv"))

    # Separar las características (X) y el target (y)
    X_test = test_data.drop(columns=['0'])
    y_test = test_data['0']

    # Cargar el modelo KNN entrenado
    knn_model = joblib.load(os.path.join(model_path, "knn_model.pkl"))

    # Cargar los parámetros de mlflow del archivo params.yaml
    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    # Inicializar la clase ClassifierModel (no necesitamos versión o semilla para evaluación)
    classifier_model = ClassifierModel(versionLabel="0.1")

    # Hacer predicciones
    y_pred = knn_model.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy,f1,precision,recall = classifier_model.evaluate_model_performance(y_test, y_pred, model_name="KNN")

    # Cargar el run_id almacenado por train.py
    with open(os.path.join(model_path, "mlflow_run_id.txt"), "r") as f:
        run_id = f.read().strip()

    # Inicializar MLFlow y registrar las métricas
    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])

    with mlflow.start_run(run_id=run_id):
        # Registrar las métricas en MLFlow
        mlflow.log_metric("accuracy", accuracy) 
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision) 
        mlflow.log_metric("recall", recall)

    # Mostrar el resultado de la evaluación
    sys.stderr.write(f"Accuracy del modelo KNN: {accuracy}\n")

if __name__ == "__main__":
    main()
