import numpy as np
import torch

from nnbma.layers.restrictable_linear import RestrictableLinear


def test_restrict_to_output_subset():
    n_dim_in, n_dim_out, n_dim_out_subset = 4, 3, 2
    list_indices_restricted = [0, 2]

    layer = RestrictableLinear(n_dim_in, n_dim_out)
    layer.eval()  # ensure not training mode

    # test shapes of weights and biases
    assert layer.weight.shape == (n_dim_out, n_dim_in)
    assert layer.bias.shape == (n_dim_out,)

    layer.restrict_to_output_subset(list_indices_restricted)
    assert layer.weight.shape == (n_dim_out, n_dim_in)
    assert layer.bias.shape == (n_dim_out,)

    assert layer.subweight.shape == (n_dim_out_subset, n_dim_in)
    assert layer.subbias.shape == (n_dim_out_subset,)


def test_forward():
    n_dim_in, n_dim_out = 4, 3
    list_indices_restricted = [0, 2]

    layer = RestrictableLinear(n_dim_in, n_dim_out)
    layer.eval()  # ensure not training mode

    N = 1000
    x = torch.normal(mean=torch.zeros((N, n_dim_in)), std=torch.ones((N, n_dim_in)))
    y_full = layer.forward(x).numpy()

    layer.restrict_to_output_subset(list_indices_restricted)

    y_small = layer.forward(x).numpy()
    assert np.max(np.abs(y_full[:, list_indices_restricted] - y_small[:, :])) == 0.0
