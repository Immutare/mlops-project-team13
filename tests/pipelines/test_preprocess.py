import numpy as np
import pytest
from pandas import DataFrame

from maternalhealth.pipeline.preprocess import saveLabelEncoder, transformDF


@pytest.fixture
def transform(prepared_dataset):
    return transformDF(prepared_dataset)


def find_outliers_IQR(df: DataFrame):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    IQR = q3 - q1
    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]

    return outliers


def test_no_outliers(transform):
    transformedDF, _ = transform
    numerics = transformedDF.copy().select_dtypes(include=[np.number])

    hasNoOutliers = True
    for column in numerics.columns:
        column_outliers = len(find_outliers_IQR(numerics[column]))
        if column_outliers != 0:
            hasNoOutliers = False
            break

    assert hasNoOutliers


def test_was_min_max(transform):
    transformedDF, _ = transform
    numerics = (
        transformedDF.copy()
        .drop(columns=["RiskLevel"])
        .select_dtypes(include=[np.number])
        .columns
    )

    isMinMaxed = True

    for column in numerics:
        # print(column)
        # print(f"Min: [{transformedDF[column].min()}] Max: [{transformedDF[column].max()}]") # noqa: E501
        if (int(transformedDF[column].min() * 100) < 0) or (
            int(transformedDF[column].max() * 100) > 100
        ):
            isMinMaxed = False

    assert isMinMaxed


def test_saves_labelencoder(transform, tmp_path):
    _, labelEncoder = transform

    tmp_directory = tmp_path / "sub"
    tmp_directory.mkdir()

    tempDir = str(tmp_directory)
    encoderFileName = "test_saves_labelencoder.pkl"
    saveLabelEncoder(labelEncoder, tempDir, encoderFileName)

    assert len(list(tmp_path.iterdir())) == 1
