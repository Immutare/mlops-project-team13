import os
import sys
import joblib
import pandas as pd
import yaml
import mlflow
from model.classifiermodel import ClassifierModel

def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py preprocessed-dir-path model-dir-path model-name\n")
        sys.exit(1)

    in_path = sys.argv[1]  # Ruta de los datos preprocesados (test)
    model_path = sys.argv[2]  # Ruta donde se guardó el modelo entrenado
    model_name = sys.argv[3]  # Nombre del modelo

    test_data = pd.read_csv(os.path.join(in_path, "test.csv"))
    X_test = test_data.drop(columns=['0'])
    y_test = test_data['0']

    # Cargar el modelo entrenado
    model = joblib.load(os.path.join(model_path, f"{model_name}.pkl"))

    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    classifier_model = ClassifierModel(versionLabel="0.1")

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy, f1, precision, recall = classifier_model.evaluate_model_performance(y_test, y_pred, model_name=model_name)

    # Cargar el run_id almacenado por train.py
    with open(os.path.join(model_path, f"mlflow_run_id_{model_name}.txt"), "r") as f:
        run_id = f.read().strip()

    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

if __name__ == "__main__":
    main()
