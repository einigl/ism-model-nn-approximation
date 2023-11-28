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


def test_standardization():
    n_dim_in = 2
    order = 3
    poly_exp = PolynomialExpansion(n_dim_in, order)

    atol = 1e-6
    batch_size = 10
    x1 = torch.normal(1, 1, size=(batch_size, n_dim_in))
    x2 = torch.normal(3, 1, size=(batch_size, n_dim_in))
    x = torch.row_stack((x1, x2))

    poly_exp.update_standardization(x1)
    y1 = poly_exp.forward(x1)
    assert torch.isclose(y1.mean(), torch.zeros_like(y1), atol=atol).all()
    assert torch.isclose(y1.std(unbiased=False), torch.ones_like(y1), atol=atol).all()

    poly_exp.update_standardization(x2)
    y2 = poly_exp.forward(x2)
    y = poly_exp.forward(x)
    assert torch.isclose(y.mean(), torch.zeros_like(y), atol=atol).all()
    assert torch.isclose(y.std(unbiased=False), torch.ones_like(y), atol=atol).all()

    poly_exp.update_standardization(x2, reset=True)
    y2 = poly_exp.forward(x2)
    assert torch.isclose(y2.mean(), torch.zeros_like(y2), atol=atol).all()
    assert torch.isclose(y2.std(unbiased=False), torch.ones_like(y2), atol=atol).all()


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
