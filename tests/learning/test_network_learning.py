import numpy as np
from torch import nn, optim

from nnbma.dataset import MaskDataset, RegressionDataset
from nnbma.learning import LearningParameters, MaskedMSELoss, learning_procedure
from nnbma.networks import FullyConnected


def _init():
    n_entries = 50
    n_input_features = 5
    n_output_features = 10

    # Dataset
    x = np.random.normal(0, 1, size=(n_entries, n_input_features))
    y = np.random.normal(0, 1, size=(n_entries, n_output_features))
    m = np.abs(np.random.normal(0, 1, size=(n_entries, n_output_features))) > 2

    dataset = RegressionDataset(x, y)
    maskset = MaskDataset(m)

    # Splitting
    idx = list(range(n_entries))
    train_idx, val_idx = idx[: round(0.8 * n_entries)], idx[round(0.8 * n_entries) :]

    # Network
    net = FullyConnected([n_input_features, 10, n_output_features], nn.ReLU())

    return dataset, maskset, train_idx, val_idx, net


def test_learning_procedure():
    dataset, _, train_idx, val_idx, net = _init()

    # Hyperparameters
    loss_fun = nn.MSELoss()
    epochs = 5
    batch_size = 10
    optimizer = optim.Adam(net.parameters())

    learning_params = LearningParameters(
        loss_fun,
        epochs,
        batch_size,
        optimizer,
    )

    # Learning procedure
    learning_procedure(
        net,
        dataset,
        learning_params,
        train_samples=train_idx,
        val_samples=val_idx,
    )


def test_masked_learning_procedure():
    dataset, maskset, train_idx, val_idx, net = _init()

    # Hyperparameters
    loss_fun = MaskedMSELoss()
    epochs = 5
    batch_size = 10
    optimizer = optim.Adam(net.parameters())

    learning_params = LearningParameters(
        loss_fun,
        epochs,
        batch_size,
        optimizer,
    )

    # Learning procedure
    learning_procedure(
        net,
        dataset,
        learning_params,
        train_samples=train_idx,
        val_samples=val_idx,
        mask_dataset=maskset,
    )
