import numpy as np

from nnbma import MaskDataset, MaskSubset


def _init() -> MaskDataset:
    n_entries = 50
    n_features = 10
    m = np.abs(np.random.normal(0, 1, size=(n_entries, n_features))) > 2
    features_names = [f"feature-{i+1}" for i in range(n_features)]
    return MaskDataset(m, features_names=features_names)


def test_attributes():
    dataset = _init()
    assert dataset.n_features == 10
    assert len(dataset) == 50
    assert dataset.features_names == [f"feature-{i+1}" for i in range(10)]
    assert dataset.features_size() == 10 * 50


def test_pandas():
    dataset = _init()
    df = dataset.to_pandas()
    assert list(df.columns) == dataset.features_names
    assert len(df) == len(dataset)
    new_dataset = dataset.from_pandas(df)
    assert np.all(dataset.getall(numpy=True) == new_dataset.getall(numpy=True))
