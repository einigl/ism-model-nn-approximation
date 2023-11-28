import numpy as np
import pytest

from nnbma import RegressionDataset, RegressionSubset


@pytest.fixture(scope="module")
def dataset() -> RegressionDataset:
    n_entries = 50
    n_input_features = 5
    n_output_features = 10
    x = np.random.normal(0, 1, size=(n_entries, n_input_features))
    y = np.random.normal(0, 1, size=(n_entries, n_output_features))
    input_names = [f"input-{i+1}" for i in range(n_input_features)]
    output_names = [f"output-{i+1}" for i in range(n_output_features)]
    return RegressionDataset(x, y, inputs_names=input_names, outputs_names=output_names)


def test_attributes(dataset: RegressionDataset):
    assert dataset.n_inputs == 5
    assert dataset.n_outputs == 10
    assert len(dataset) == 50
    assert dataset.inputs_names == [f"input-{i+1}" for i in range(5)]
    assert dataset.outputs_names == [f"output-{i+1}" for i in range(10)]
    assert dataset.inputs_size() == 5 * 50
    assert dataset.outputs_size() == 10 * 50


def test_non_finite(dataset: RegressionDataset):
    x, y = dataset.getall(numpy=True)

    assert not any(dataset.has_nan())
    assert not any(dataset.has_nonfinite())

    _x, _y = x.copy(), y.copy()
    _x[np.random.randint(x.shape[0]), np.random.randint(x.shape[1])] = np.nan
    _y[np.random.randint(y.shape[0]), np.random.randint(y.shape[1])] = np.inf

    _dataset = RegressionDataset(_x, _y)

    assert _dataset.has_nan()[0] and not _dataset.has_nan()[1]
    assert _dataset.has_nonfinite()[0] and _dataset.has_nonfinite()[1]


def test_pandas(dataset: RegressionDataset):
    df_in, df_out = dataset.to_pandas()
    assert list(df_in.columns) == dataset.inputs_names
    assert list(df_out.columns) == dataset.outputs_names
    assert len(df_in) == len(dataset)
    assert len(df_out) == len(dataset)
    new_dataset = dataset.from_pandas(df_in, df_out)
    assert np.all(
        np.isclose(dataset.getall(numpy=True)[0], new_dataset.getall(numpy=True)[0])
    )
    assert np.all(
        np.isclose(dataset.getall(numpy=True)[1], new_dataset.getall(numpy=True)[1])
    )
