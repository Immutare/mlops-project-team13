import os

import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def transformed_dataset():
    path = "src/maternalhealth/data/preprocessed"
    filename = "train.csv"
    filepath = os.path.join(path, filename)
    dataframe = pd.read_csv(filepath, encoding="utf-8")
    return dataframe


@pytest.fixture(scope="session", autouse=True)
def raw_dataset():
    path = "src/maternalhealth/dataset"
    filename = "MaternalHealthRiskDataSet.csv"
    filepath = os.path.join(path, filename)
    dataframe = pd.read_csv(filepath, encoding="utf-8")
    return dataframe


@pytest.fixture(scope="session", autouse=True)
def prepared_dataset():
    path = "src/maternalhealth/data/prepared"
    filename = "train.csv"
    filepath = os.path.join(path, filename)
    dataframe = pd.read_csv(filepath, encoding="utf-8")
    return dataframe


@pytest.fixture(scope="session", autouse=True)
def test_transformed_dataset():
    path = "src/maternalhealth/data/preprocessed"
    filename = "test.csv"
    filepath = os.path.join(path, filename)
    dataframe = pd.read_csv(filepath, encoding="utf-8")
    return dataframe
