import numpy as np
import pytest
from torch import nn, optim

from nnbma.dataset import MaskDataset, MaskSubset, RegressionDataset, RegressionSubset
from nnbma.learning import (
    LearningParameters,
    LinearBatchScheduler,
    MaskedMSELoss,
    absolute_error,
    learning_procedure,
    relative_error,
)
from nnbma.networks import FullyConnected


@pytest.fixture(scope="module")
def init():
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


def test_learning_procedure(init: tuple):
    dataset, _, train_idx, val_idx, net = init

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


def test_learning_procedure_wo_splitting(init: tuple):
    dataset, _, train_idx, val_idx, net = init

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

    train_set = RegressionSubset(dataset, train_idx)
    val_set = RegressionSubset(dataset, val_idx)

    # Learning procedure
    learning_procedure(
        net,
        (train_set, val_set),
        learning_params,
    )


def test_masked_learning_procedure(init: tuple):
    dataset, maskset, train_idx, val_idx, net = init

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


def test_masked_learning_procedure_wo_splitting(init: tuple):
    dataset, maskset, train_idx, val_idx, net = init

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

    train_set = RegressionSubset(dataset, train_idx)
    val_set = RegressionSubset(dataset, val_idx)

    train_maskset = MaskSubset(maskset, train_idx)
    val_maskset = MaskSubset(maskset, val_idx)

    # Learning procedure
    learning_procedure(
        net,
        (train_set, val_set),
        learning_params,
        train_samples=train_idx,
        val_samples=val_idx,
        mask_dataset=(train_maskset, val_maskset),
    )


def test_learning_procedure_batch_scheduler(init: tuple):
    dataset, _, train_idx, val_idx, net = init

    # Hyperparameters
    loss_fun = nn.MSELoss()
    epochs = 5
    batch_size = LinearBatchScheduler(10, 50, epochs)
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


def test_learning_procedure_metrics(init: tuple):
    dataset, _, train_idx, val_idx, net = init

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
        additional_metrics={
            "rel": relative_error,
            "abs": absolute_error,
        },
    )
