import pandas as pd
import pytest

from maternalhealth.pipeline.prepare import split_test_train


def get_dataset():
    return pd.read_csv("src/maternalhealth/dataset/MaternalHealthRiskDataSet.csv")


@pytest.mark.parametrize(
    "train_split,seed",
    [
        (0.8, 42),
        (0.5, 42),
        (0.3, 42),
        (0.2, 42),
    ],
)
def test_split_test_train(train_split, seed):
    dataframe = get_dataset()
    train, test = split_test_train(dataframe, train_split, seed)

    split_train_total = len(dataframe) * train_split

    assert (len(train) > split_train_total - 1) and (len(train) < split_train_total + 1)
