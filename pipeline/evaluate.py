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

    in_path = sys.argv[1]
    model_path = sys.argv[2]

    test_data = pd.read_csv(os.path.join(in_path, "test.csv"))
    X_test = test_data.drop(columns=['0'])
    y_test = test_data['0']

    knn_model = joblib.load(os.path.join(model_path, "knn_model.pkl"))
    logistic_model = joblib.load(os.path.join(model_path, "logistic_model.pkl"))

    mlflow_params = yaml.safe_load(open("params.yaml"))["mlflow"]

    classifier_model = ClassifierModel(versionLabel="0.1")

    y_pred_knn = knn_model.predict(X_test)
    accuracy_knn, f1_knn, precision_knn, recall_knn = classifier_model.evaluate_model_performance(y_test, y_pred_knn, model_name="KNN")

    y_pred_logistic = logistic_model.predict(X_test)
    accuracy_logistic, f1_logistic, precision_logistic, recall_logistic = classifier_model.evaluate_model_performance(y_test, y_pred_logistic, model_name="Logistic")

    with open(os.path.join(model_path, "mlflow_run_id.txt"), "r") as f:
        run_id = f.read().strip()

    mlflow.set_experiment(mlflow_params['experiment_name'])
    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy_knn", accuracy_knn)
        mlflow.log_metric("f1_score_knn", f1_knn)
        mlflow.log_metric("precision_knn", precision_knn)
        mlflow.log_metric("recall_knn", recall_knn)

        mlflow.log_metric("accuracy_logistic", accuracy_logistic)
        mlflow.log_metric("f1_score_logistic", f1_logistic)
        mlflow.log_metric("precision_logistic", precision_logistic)
        mlflow.log_metric("recall_logistic", recall_logistic)

    sys.stderr.write(f"Accuracy del modelo KNN: {accuracy_knn}\n")
    sys.stderr.write(f"Accuracy del modelo Logístico: {accuracy_logistic}\n")

if __name__ == "__main__":
    main()
