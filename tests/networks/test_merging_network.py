import os
import shutil

import torch
from torch import nn
from torch.func import jacfwd, jacrev, vmap

from nnbma import FullyConnected, MergingNetwork


def _init() -> MergingNetwork:
    layers_sizes = [5, 10, 10, 5]
    activation = nn.ELU()
    subnet1 = FullyConnected(layers_sizes, activation)

    layers_sizes = [5, 10, 10, 15]
    activation = nn.ELU()
    subnet2 = FullyConnected(layers_sizes, activation)

    net = MergingNetwork([subnet1, subnet2])

    return net


def test_shape():
    net = _init()

    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))
    y = net.forward(x)

    assert y.shape == (batch_size, net.output_features)


def test_save_load():
    net = _init()
    path = os.path.dirname(os.path.abspath(__file__))

    net.save("temp-net", path)
    net2 = MergingNetwork.load("temp-net", path)
    shutil.rmtree(os.path.join(path, "temp-net"))

    x = torch.normal(0, 1, size=(net.input_features,))
    assert torch.all(net(x) == net2(x))


# def test_verify():
#     net = _init()
#     path = os.path.dirname(os.path.abspath(__file__))

#     net.save("temp-verify-merging", path)


def test_derivatives():
    net = _init()
    path = os.path.dirname(os.path.abspath(__file__))

    batch_size = 50
    x = torch.normal(0, 1, size=(batch_size, net.input_features))

    dy_rev = vmap(jacrev(net.forward))(x)
    dy_fwd = vmap(jacfwd(net.forward))(x)

    assert dy_rev.shape == (batch_size, net.output_features, net.input_features)
    assert dy_fwd.shape == (batch_size, net.output_features, net.input_features)

    assert torch.any(dy_rev)
    assert torch.any(dy_fwd)
