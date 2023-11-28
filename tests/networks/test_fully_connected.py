import os
import shutil

import pytest
import torch
from torch import nn
from torch.func import jacfwd, jacrev, vmap

from nnbma import FullyConnected


@pytest.fixture(scope="module")
def net() -> FullyConnected:
    layers_sizes = [5, 10, 10, 20]
    activation = nn.ELU()
    net = FullyConnected(
        layers_sizes,
        activation,
    )
    return net


def test_shape(net: FullyConnected):
    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))
    y = net.forward(x)

    assert y.shape == (batch_size, net.output_features)


def test_save_load(net: FullyConnected):
    path = os.path.dirname(os.path.abspath(__file__))

    net.save("temp-net", path)
    net2 = FullyConnected.load("temp-net", path)
    shutil.rmtree(os.path.join(path, "temp-net"))

    x = torch.normal(0, 1, size=(net.input_features,))
    assert torch.all(net(x) == net2(x))


def test_derivatives(net: FullyConnected):
    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))

    dy_rev = vmap(jacrev(net.forward))(x)
    dy_fwd = vmap(jacfwd(net.forward))(x)

    assert dy_rev.shape == (batch_size, net.output_features, net.input_features)
    assert dy_fwd.shape == (batch_size, net.output_features, net.input_features)

    assert torch.any(dy_rev)
    assert torch.any(dy_fwd)
