import os
import random
import shutil

import pytest
import torch
from torch import nn
from torch.func import jacfwd, jacrev, vmap

from nnbma import FullyConnected, MergingNetwork


def _init(names) -> MergingNetwork:
    layers_sizes = [5, 10, 10, 5]
    activation = nn.ELU()
    names1 = [f"output-1-{i}" for i in range(layers_sizes[-1])] if names else None
    subnet1 = FullyConnected(layers_sizes, activation, outputs_names=names1)

    layers_sizes = [5, 10, 10, 15]
    activation = nn.ELU()
    names2 = [f"output-2-{i}" for i in range(layers_sizes[-1])] if names else None
    subnet2 = FullyConnected(layers_sizes, activation, outputs_names=names2)

    names = names1 + names2
    random.seed(0)
    random.shuffle(names)
    net = MergingNetwork([subnet1, subnet2], names)

    return net


@pytest.fixture(scope="module")
def net() -> MergingNetwork:
    return _init(names=True)


def test_shape(net: MergingNetwork):
    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))
    y = net.forward(x)

    assert y.shape == (batch_size, net.output_features)


def test_restrict_to_output_subset(net: MergingNetwork):
    net.restrict_to_output_subset(None)  # TODO
    # net.restrict_to_output_subset(['output-1-0', 'output-2-1', 'output-2-1'])

    # x = torch.normal(0, 1, size=(1, net.input_features))
    # y = net.forward(x)
    # assert y.shape == (1, 2)
    # assert torch.isclose(y[:, 1], y[:, 2])

    # net.restrict_to_output_subset(None)

    # y = net.forward(x)
    # assert y.shape(1, net.output_features)


def test_save_load(net: MergingNetwork):
    path = os.path.dirname(os.path.abspath(__file__))

    net.save("temp-net", path)
    net2 = MergingNetwork.load("temp-net", path)
    shutil.rmtree(os.path.join(path, "temp-net"))

    x = torch.normal(0, 1, size=(net.input_features,))
    assert torch.all(net(x) == net2(x))


def test_derivatives(net: MergingNetwork):
    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))

    dy_rev = vmap(jacrev(net.forward))(x)
    dy_fwd = vmap(jacfwd(net.forward))(x)

    assert dy_rev.shape == (batch_size, net.output_features, net.input_features)
    assert dy_fwd.shape == (batch_size, net.output_features, net.input_features)

    assert torch.any(dy_rev)
    assert torch.any(dy_fwd)
