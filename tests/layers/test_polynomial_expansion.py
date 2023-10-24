import numpy as np
import torch

from nnbma.layers.polynomial_expansion import PolynomialExpansion


def test_expanded_features():
    n_dim_in = 3
    order = 2
    n_expanded = PolynomialExpansion.expanded_features(order, n_dim_in)

    assert n_expanded == 9


def test_forward():
    n_dim_in = 2
    order = 3
    poly_exp = PolynomialExpansion(n_dim_in, order)

    x = torch.ones((1, n_dim_in))
    y = poly_exp.forward(x)

    assert y.shape == (1, 9)
    assert np.allclose(y, torch.ones((1, 9)))
