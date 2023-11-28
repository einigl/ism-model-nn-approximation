import numpy as np
import pandas as pd
import pytest

from nnbma.operators import InverseNormalizer, Normalizer, NormTypes

atol = 1e-6


@pytest.fixture(scope="module")
def x() -> np.ndarray:
    mus = np.array([1, 2, 3])
    sigmas = np.array([1, 2, 3])
    n_samples = 20
    return np.random.normal(mus, sigmas, size=(n_samples, mus.size))


def test_none(x: np.ndarray):
    df = pd.DataFrame(x)

    op = Normalizer(df, norm_type=NormTypes.NONE)
    y = op(x)
    assert np.all(y == x)

    inv_op = InverseNormalizer(df, norm_type=NormTypes.NONE)
    x_hat = inv_op(y)
    assert np.all(x_hat == x)


def test_mean0(x: np.ndarray):
    df = pd.DataFrame(x)
    n_features = x.shape[-1]

    op = Normalizer(df, norm_type=NormTypes.MEAN0)
    y = op(x)
    assert np.all(np.isclose(np.mean(y, 0), np.zeros(n_features)))

    inv_op = InverseNormalizer(df, norm_type=NormTypes.MEAN0)
    x_hat = inv_op(y)
    assert np.all(np.isclose(x_hat, x))


def test_std1(x: np.ndarray):
    df = pd.DataFrame(x)
    n_features = x.shape[-1]

    op = Normalizer(df, norm_type=NormTypes.STD1)
    y = op(x)
    assert np.all(np.isclose(np.std(y, 0, ddof=1), np.ones(n_features)))

    inv_op = InverseNormalizer(df, norm_type=NormTypes.STD1)
    x_hat = inv_op(y)
    assert np.all(np.isclose(x_hat, x))


def test_mean0std1(x: np.ndarray):
    df = pd.DataFrame(x)
    n_features = x.shape[-1]

    op = Normalizer(df, norm_type=NormTypes.MEAN0STD1)
    y = op(x)
    assert np.all(np.isclose(np.mean(y, 0), np.zeros(n_features)))
    assert np.all(np.isclose(np.std(y, 0, ddof=1), np.ones(n_features)))

    inv_op = InverseNormalizer(df, norm_type=NormTypes.MEAN0STD1)
    x_hat = inv_op(y)
    assert np.all(np.isclose(x_hat, x))


def test_min0max1(x: np.ndarray):
    df = pd.DataFrame(x)
    n_features = x.shape[-1]

    op = Normalizer(df, norm_type=NormTypes.MIN0MAX1)
    y = op(x)
    assert np.all(np.isclose(np.min(y, 0), np.zeros(n_features)))
    assert np.all(np.isclose(np.max(y, 0), np.ones(n_features)))

    inv_op = InverseNormalizer(df, norm_type=NormTypes.MIN0MAX1)
    x_hat = inv_op(y)
    assert np.all(np.isclose(x_hat, x))


def test_min1max1(x: np.ndarray):
    df = pd.DataFrame(x)
    n_features = x.shape[-1]

    op = Normalizer(df, norm_type=NormTypes.MIN1MAX1)
    y = op(x)
    assert np.all(np.isclose(np.min(y, 0), -np.ones(n_features)))
    assert np.all(np.isclose(np.max(y, 0), np.ones(n_features)))

    inv_op = InverseNormalizer(df, norm_type=NormTypes.MIN1MAX1)
    x_hat = inv_op(y)
    assert np.all(np.isclose(x_hat, x))
