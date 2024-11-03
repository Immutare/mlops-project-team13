import mlflow
import yaml

# @pytest.mark.parametrize(
#     "model_name",
#     [
#         ("decision_tree_model"),
#         ("knn_model"),
#         ("logistic_model"),
#         ("random_forest_model"),
#         ("svm_model"),
#         ("xgboost_model"),
#     ],
# )


def retrieve_mlflow_params():
    return yaml.safe_load(open("src/maternalhealth/params.yaml"))["mlflow"]


def test_best_model(test_transformed_dataset):
    mlflow_params = retrieve_mlflow_params()

    client = mlflow.tracking.MlflowClient(
        mlflow_params.get("tracking_uri", "http://127.0.0.1:5000")
    )

    experiment_name = mlflow_params.get(
        "experiment_name", "MaternalHealthRisk"
    )

    current_experiment = dict(client.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment["experiment_id"]

    runs = client.search_runs(
        experiment_id, "", order_by=["metrics.accuracy DESC"]
    )
    best_run = runs[0]

    best_accuracy = best_run.data.metrics.get("accuracy", 0)

    # X_test = test_transformed_dataset.drop(columns=["RiskLevel"])
    # y_test = test_transformed_dataset["RiskLevel"]

    # # Hacer predicciones
    # y_pred = best_run.predict(X_test)

    # classifier_model = ClassifierModel()
    # accuracy, _, _, _ = (
    #     classifier_model.evaluate_model_performance(  # noqa: E501
    #         y_test, y_pred, model_name="BestModel"
    #     )
    # )

    assert best_accuracy >= 0.8
