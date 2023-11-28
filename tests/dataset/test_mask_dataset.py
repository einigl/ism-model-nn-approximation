import numpy as np
import pytest

from nnbma import MaskDataset, MaskSubset


@pytest.fixture(scope="module")
def maskset() -> MaskDataset:
    n_entries = 50
    n_features = 10
    m = np.abs(np.random.normal(0, 1, size=(n_entries, n_features))) > 2
    features_names = [f"feature-{i+1}" for i in range(n_features)]
    return MaskDataset(m, features_names=features_names)


def test_attributes(maskset: MaskDataset):
    assert maskset.n_features == 10
    assert len(maskset) == 50
    assert maskset.features_names == [f"feature-{i+1}" for i in range(10)]
    assert maskset.features_size() == 10 * 50


def test_pandas(maskset: MaskDataset):
    df = maskset.to_pandas()
    assert list(df.columns) == maskset.features_names
    assert len(df) == len(maskset)
    new_maskset = maskset.from_pandas(df)
    assert np.all(maskset.getall(numpy=True) == new_maskset.getall(numpy=True))
