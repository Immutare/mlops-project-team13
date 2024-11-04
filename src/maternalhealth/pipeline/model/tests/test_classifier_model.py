"""
Unit tests for the ClassifierModel class,
including verification of initialization,
property setting, and data
cleaning/preprocessing.
"""

import numpy as np
import pandas as pd
import pytest

from maternalhealth.pipeline.model.classifiermodel import ClassifierModel


@pytest.fixture
def mock_model(mocker):
    """Fixture to create a mock model."""
    return mocker.Mock()


class TestClassifierModel:
    """Tests for ClassifierModel, including initialization, properties,
    and data preprocessing."""

    @pytest.fixture
    def classifier_model(self):
        """Fixture to initialize ClassifierModel for testing."""
        return ClassifierModel(versionLabel="v1")

    def test_initialization(self, classifier_model):
        """Test that ClassifierModel initializes with the correct
        attributes."""
        assert classifier_model._model is None
        assert classifier_model.IQ_LOWER_RANGE == 1.5
        assert classifier_model.RANDOM_STATE_SEED == 42
        assert classifier_model.MODEL_DUMP_PATH == "dump/"

    def test_property_setters(self, classifier_model):
        """Test that property setters correctly update values."""
        classifier_model.model = "TestModel"
        classifier_model.IQ_LOWER_RANGE = 1.2
        classifier_model.label_encoder_filename = "test_label_encoder.pkl"

        assert classifier_model.model == "TestModel"
        assert classifier_model.IQ_LOWER_RANGE == 1.2
        assert (
            classifier_model.label_encoder_filename == "test_label_encoder.pkl"
        )

    def test_predict_with_model(self, classifier_model, mock_model):
        """Test the predict method when a model is loaded."""
        classifier_model._model = mock_model

        X_test = pd.DataFrame(
            {
                "Age": [30, 40],
                "SystolicBP": [120, 140],
                "DiastolicBP": [80, 90],
                "BS": [0.5, 0.6],
                "BodyTemp": [98.6, 99.1],
                "HeartRate": [70, 80],
            }
        )

        mock_model.predict.return_value = np.array([0, 1])

        predictions = classifier_model.predict(X_test)
        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_predict_without_model(self, classifier_model):
        """Test the predict method raises an exception when
        no model is loaded."""
        with pytest.raises(Exception, match="Couldn't find a model"):
            classifier_model.predict(pd.DataFrame())

    def test_get_outliers(self, classifier_model):
        """Test the getOutliers method."""
        X_test = pd.DataFrame(
            {
                "Age": [25, 35, 29, 30, 40],
                "SystolicBP": [130, 140, 90, 140, 150],
                "DiastolicBP": [80, 90, 70, 85, 88],
                "BS": [15, 13, 8, 7, 5],
                "BodyTemp": [98, 98, 100, 98, 99],
                "HeartRate": [86, 70, 80, 70, 75],
            }
        )

        classifier_model.IQ_LOWER_RANGE = 1.5

        outliers = classifier_model.getOutliers(X_test)

        assert outliers.shape == X_test.shape

        assert outliers.dtypes.apply(lambda x: x == np.bool_).all()

    def test_clean_and_preprocess_data(self, classifier_model):
        """Test the cleanAndPreprocessData method."""
        X_test = pd.DataFrame(
            {
                "Age": [25, 35, 29, 30],
                "SystolicBP": [130, 140, 90, 140],
                "DiastolicBP": [80, 90, 70, 85],
                "BS": [15, 13, 8, 7],
                "BodyTemp": [98, 98, 100, 98],
                "HeartRate": [86, 70, 80, 70],
            }
        )
        y_test = pd.Series(
            ["high risk", "high risk", "high risk", "high risk"]
        )

        X_normalized, y_encoded = classifier_model.cleanAndPreprocessData(
            X_test, y_test
        )

        assert X_normalized.shape == (4, 6)
        assert len(y_encoded) == 4

        assert X_normalized["HeartRate"].isnull().sum() == 0
        assert X_normalized["Age"].isnull().sum() == 0

    def test_split_train_test_val(self, classifier_model):
        """Test the splitTrainTestVal method."""
        X_test = pd.DataFrame(
            {
                "Age": [25, 35, 29, 30, 40],
                "SystolicBP": [130, 140, 90, 140, 150],
                "DiastolicBP": [80, 90, 70, 85, 88],
                "BS": [15, 13, 8, 7, 5],
                "BodyTemp": [98, 98, 100, 98, 99],
                "HeartRate": [86, 70, 80, 70, 75],
            }
        )
        y_test = pd.Series(
            ["high risk", "high risk", "high risk", "mid risk", "low risk"]
        )

        Xtrain, ytrain, Xtest, ytest, Xval, yval = (
            classifier_model.splitTrainTestVal(X_test, y_test)
        )

        assert (
            Xtrain.shape[0] + Xtest.shape[0] + Xval.shape[0] == X_test.shape[0]
        )
        assert (
            ytrain.shape[0] + ytest.shape[0] + yval.shape[0] == y_test.shape[0]
        )

        assert Xtrain.shape[0] > 0
        assert Xtest.shape[0] > 0
        assert Xval.shape[0] > 0

    def test_label_encoding(self, classifier_model):
        """Test the labelEncoding method."""
        Y_test = pd.Series(["high risk", "mid risk", "low risk", "high risk"])
        Y_encoded = classifier_model.labelEncoding(Y_test)
        newdat = [item for item in Y_encoded]
        np.testing.assert_array_equal(
            newdat[0],
            [
                0,
                2,
                1,
                0,
            ],
        )
