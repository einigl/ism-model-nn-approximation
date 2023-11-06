import torch
from torch.func import jacrev, vmap

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

    n_poly = 9
    assert poly_exp.n_expanded_features == n_poly
    assert y.shape == (1, n_poly)
    assert torch.allclose(y, torch.ones((1, n_poly)))


def test_diff():
    n_dim_in = 2
    order = 2
    poly_exp = PolynomialExpansion(n_dim_in, order)
    dpoly_exp = vmap(jacrev(poly_exp))

    x = torch.tensor([[1, 2]], dtype=torch.float32)
    dy = dpoly_exp(x).squeeze(0)

    assert torch.allclose(
        dy, torch.tensor([[1, 0], [0, 1], [2, 0], [2, 1], [0, 4]], dtype=dy.dtype)
    )
