import os
import shutil

import pytest
import torch
from torch import nn
from torch.func import jacfwd, jacrev, vmap

from nnbma import DenselyConnected


@pytest.fixture(scope="module")
def net() -> DenselyConnected:
    input_features, output_features = 5, 20
    n_layers = 10
    growing_factor = 0.2
    activation = nn.ELU()
    net = DenselyConnected(
        input_features, output_features, n_layers, growing_factor, activation
    )
    return net


def test_shape(net: DenselyConnected):
    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))
    y = net.forward(x)

    assert y.shape == (batch_size, net.output_features)


def test_layers(net: DenselyConnected):
    assert net.layers_sizes == [5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 20]


def test_save_load(net: DenselyConnected):
    path = os.path.dirname(os.path.abspath(__file__))

    net.save("temp-net", path)
    net2 = DenselyConnected.load("temp-net", path)
    shutil.rmtree(os.path.join(path, "temp-net"))

    x = torch.normal(0, 1, size=(net.input_features,))
    assert torch.all(net(x) == net2(x))


def test_derivatives(net: DenselyConnected):
    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))

    dy_rev = vmap(jacrev(net.forward))(x)
    dy_fwd = vmap(jacfwd(net.forward))(x)

    assert dy_rev.shape == (batch_size, net.output_features, net.input_features)
    assert dy_fwd.shape == (batch_size, net.output_features, net.input_features)

    assert torch.any(dy_rev)
    assert torch.any(dy_fwd)
