import os
import shutil

import numpy as np
import pytest
from torch import nn, normal

from nnbma.networks import FullyConnected
from nnbma.operators import log10, pow10


@pytest.fixture(scope="module")
def net() -> FullyConnected:
    return FullyConnected(
        [2, 5, 10], nn.ReLU(), inputs_transformer=log10, outputs_transformer=pow10
    )


def test_evaluate(net: FullyConnected):
    x = np.random.exponential(1, size=(1, net.input_features))

    y0 = net.evaluate(x)
    y1 = net.evaluate(x, transform_inputs=True)
    y2 = net.evaluate(x, transform_outputs=True)

    assert np.any(~np.isclose(y0, y1, atol=1e-6))
    assert np.all(y2 >= 0)


def test_restrict_to_output_subset():
    n_inputs, n_outputs = 2, 10
    net = FullyConnected(
        [2, 5, 10],
        nn.ReLU(),
        last_restrictable=False,
        outputs_names=[f"output-{i}" for i in range(n_outputs)],
    )

    x = np.random.normal(0, 1, size=n_inputs)

    net.restrict_to_output_subset([f"output-{3}", f"output-{8}", f"output-{8}"])
    y1 = net(x)
    assert y1.shape == (3,)

    net.restrict_to_output_subset([3, 8, 8])
    y2 = net(x)
    assert y2.shape == (3,)

    assert np.all(np.isclose(y1, y2))

    net.restrict_to_output_subset(None)
    y = net(x)
    assert y.shape == (n_outputs,)


def test_copy(net: FullyConnected):
    netbis = net.copy()

    x = normal(0, 1, size=(1, net.input_features))
    y = net.forward(x)
    ybis = netbis.forward(x)

    assert (y == ybis).all()


def test_save_if_exists(net: FullyConnected):
    path = os.path.dirname(os.path.abspath(__file__))

    net.save("temp-net", path)
    net.save("temp-net", path, overwrite=True)

    with pytest.raises(FileExistsError):
        net.save("temp-net", path, overwrite=False)
    shutil.rmtree(os.path.join(path, "temp-net"))


def test_count_parameters(net: FullyConnected):
    out = net.count_parameters()
    assert isinstance(out, int)


def test_count_bytes(net: FullyConnected):
    out = net.count_bytes()
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], float) and isinstance(out[1], str)

    out = net.count_bytes(display=True)
    assert isinstance(out, str)


def test_time(net: FullyConnected):
    out = net.time(5, 2)
    assert isinstance(out, tuple) and len(out) == 3
    assert 0 <= out[1] <= out[0] <= out[2]
