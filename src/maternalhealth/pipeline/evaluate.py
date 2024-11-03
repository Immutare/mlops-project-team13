import os
import sys

import joblib
import mlflow
import pandas as pd
import yaml

from maternalhealth.pipeline.model.classifiermodel import ClassifierModel


def load_testdf(path, filename="test.csv"):
    test_data = pd.read_csv(os.path.join(path, filename))
    X_test = test_data.drop(columns=["RiskLevel"])
    y_test = test_data["RiskLevel"]

    return X_test, y_test


def load_model(path, filename):
    return joblib.load(os.path.join(path, f"{filename}.pkl"))


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            "\tpython evaluate.py preprocessed-dir-path model-dir-path model-name\n"  # noqa: E501
        )
        sys.exit(1)

    in_path = sys.argv[1]  # Ruta de los datos preprocesados (test)
    model_path = sys.argv[2]  # Ruta donde se guardó el modelo entrenado
    model_name = sys.argv[3]  # Nombre del modelo

    X_test, y_test = load_testdf(in_path)

    # Cargar el modelo entrenado
    model = load_model(model_path, model_name)

    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    classifier_model = ClassifierModel(versionLabel="0.1")

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy, f1, precision, recall = (
        classifier_model.evaluate_model_performance(  # noqa: E501
            y_test, y_pred, model_name=model_name
        )
    )

    # Cargar el run_id almacenado por train.py
    with open(
        os.path.join(model_path, f"mlflow_run_id_{model_name}.txt"), "r"
    ) as f:
        run_id = f.read().strip()

    mlflow.set_experiment(mlflow_params["experiment_name"])
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)


if __name__ == "__main__":
    main()
