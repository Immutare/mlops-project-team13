import pytest

from maternalhealth.pipeline.prepare import split_test_train


@pytest.mark.parametrize(
    "train_split,seed",
    [
        (0.8, 42),
        (0.5, 42),
        (0.3, 42),
        (0.2, 42),
    ],
)
def test_split_test_train(train_split, seed, raw_dataset):
    train, _ = split_test_train(raw_dataset, train_split, seed)

    split_train_total = len(raw_dataset) * train_split

    assert (len(train) > split_train_total - 1) and (
        len(train) < split_train_total + 1
    )
