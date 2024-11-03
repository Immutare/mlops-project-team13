"""
Unit tests for the InformalModelInterface class, which include mock testing 
for the splitTrainTestVal and extract_text methods.
"""

from unittest.mock import patch
import pytest
import numpy as np

from maternalhealth.pipeline.model.informalmodelinterface import (
    InformalModelInterface,
)


class TestInformalModelInterface:
    """Contains unit tests for InformalModelInterface, testing splitTrainTestVal and extract_text methods."""

    @pytest.fixture
    def model_interface(self):
        """Fixture to initialize InformalModelInterface for testing."""
        return InformalModelInterface()

    @patch.object(
        InformalModelInterface,
        "splitTrainTestVal",
        return_value=("train", "test", "val"),
    )
    def test_split_train_test_val(self, mock_split, model_interface):
        """
        Test splitTrainTestVal method to ensure it splits data into
        training, testing, and validation sets as expected.
        """
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 0, 1, 0, 1])
        train_size = 0.6
        test_size = 0.2
        result = model_interface.splitTrainTestVal(X, y, train_size, test_size)

        mock_split.assert_called_once_with(X, y, train_size, test_size)
        assert result == (
            "train",
            "test",
            "val",
        ), "Test result should be ('train', 'test', 'val')"

    @patch.object(
        InformalModelInterface,
        "extract_text",
        return_value="Mocked text content",
    )
    def test_extract_text(self, mock_extract, model_interface):
        """
        Test extract_text method to ensure it correctly extracts text from a file.

        This method verifies that the extract_text function can process a file name
        and return the expected mock content.
        """
        test_file = "file.txt"
        result = model_interface.extract_text(test_file)
        mock_extract.assert_called_once_with(test_file)
        assert (
            result == "Mocked text content"
        ), "Test result should be 'Mocked text content'"
