import datetime
import random
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nnbma.dataset import MaskDataset, MaskSubset, RegressionDataset, RegressionSubset
from nnbma.networks import NeuralNetwork

from .batch_scheduler import BatchScheduler
from .loss_functions import MaskedLossFunction, MaskOverlay

__all__ = [
    "LearningParameters",
    "learning_procedure",
]


class LearningParameters:
    r"""Specifies the main parameters training, including the loss function to minimize and the stochastic gradient descent strategy."""

    def __init__(
        self,
        loss_fun: Union[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            MaskedLossFunction,
        ],
        epochs: int,
        batch_size: Union[int, BatchScheduler, None],
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
    ):
        r"""

        Parameters
        ----------
        loss_fun : Union[ Callable[[torch.Tensor, torch.Tensor], torch.Tensor], MaskedLossFunction, ]
            loss function.
        epochs : int
            total number of epochs to perform.
        batch_size : Union[int, BatchScheduler, None]
            batch size value or scheduler to use during training.
        optimizer : Optimizer
            optimizer to use for training.
        scheduler : Optional[_LRScheduler], optional
            learning rate scheduler, by default None
        """
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        if scheduler is None:
            self.scheduler = ConstantLR(optimizer, 1.0)
        else:
            self.scheduler = scheduler

    def __str__(self):
        s = "Learning parameters:\n"
        s += f"\tLoss function: {self.loss_fun}\n"
        s += f"\tEpochs: {self.epochs}\n"
        s += f"\tBatch size: {self.batch_size}\n"
        s += f"\tOptimizer: {self.optimizer}\n"
        s += f"\tScheduler: {self.scheduler}"
        return s


def learning_procedure(
    model: NeuralNetwork,
    dataset: Union[RegressionDataset, Tuple[RegressionDataset, RegressionDataset]],
    learning_parameters: Union[LearningParameters, List[LearningParameters]],
    mask_dataset: Union[
        MaskDataset, Tuple[MaskDataset, MaskDataset], None, Tuple[None, None]
    ] = None,
    train_samples: Optional[Sequence] = None,
    val_samples: Optional[Sequence] = None,
    val_frac: Optional[float] = None,
    additional_metrics: Optional[
        Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    ] = None,
    verbose_level: Literal[None, 0, 1, 2] = 1,
    seed: Optional[int] = None,
    max_iter_no_improve: Optional[int] = None,
) -> Dict[str, object]:
    r"""Performs the training of the neural network ``model`` to fit the provided training data.

    Parameters
    ----------
    model : NeuralNetwork
        model to train.
    dataset : Union[RegressionDataset, Tuple[RegressionDataset, RegressionDataset]]
        dataset to use for training and validation. This argument is used with ``mask_dataset`` (to define the corresponding masked values) and ``val_frac`` (to define the proportion of entries to use in the validation set).
    learning_parameters : Union[LearningParameters, List[LearningParameters]]
        parameters of the stochastic gradient descent algorithm.
    mask_dataset : Union[ MaskDataset, Tuple[MaskDataset, MaskDataset], None, Tuple[None, None] ], optional
        _description_, by default None
    train_samples : Optional[Sequence], optional
        samples to use for training. When used, the arguments ``dataset``, ``mask_dataset`` and ``val_frac`` are disregarded. By default None.
    val_samples : Optional[Sequence], optional
        samples to use for validation, by default None.
    val_frac : Optional[float], optional
        proportion of elements of the ``dataset`` to use in the validation set. If specified, should be between 0 and 1. By default None.
    additional_metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]], optional
        metrics to track in addition to the loss function, by default None
    verbose_level : Literal[None, 0, 1, 2], optional
        amount of information provided during training. 0 or None: no display. 1: display of the epoch bar. 2: display of network description and also epoch and batch bars. Default: 1.
    seed : Optional[int], optional
        random seed for reproducibility, by default None.
    max_iter_no_improve : Optional[int], optional
        early stopping parameter. The training stops when the loss on the validation set does not decrease for ``max_iter_no_improve`` steps. By default None.

    Returns
    -------
    Dict[str, object]
        This dictionary contains:
        - "train_loss": the time series of the average train loss per epoch.
        - "val_loss": the time series of the validation loss per epoch.
        - "train_metrics": the time series of additional metrics on train set.
        - "val_metrics": the time series of additional metrics on test set.
        - "train_set": train_set.
        - "val_set": val_set.
        - "lr": the time series of the learning rate per epoch.
        - "batch_size": the time series of the batch size per epoch.
        - "duration": total duration of training.

    Raises
    ------
    TypeError
        The ``dataset`` argument must be an instance of "RegressionDataset" or a tuple of two "RegressionDataset".
    TypeError
        The ``mask_dataset`` argument must be an instance of "MaskDataset" or a tuple of two "MaskDataset" or None.
    ValueError
        The ``mask_dataset`` argument must not be a tuple when the ``dataset`` argument is a "RegressionDataset".
    ValueError
        The ``dataset`` argument must not be a tuple when ``mask_dataset`` is a "MaskDataset".
    ValueError
        The train dataset and validation dataset must not share samples.
    ValueError
        The training dataset must not contain non finite values.
    ValueError
        The ``learning_parameters`` argument must be an instance of "LearningParameters" or a list of "LearningParameters".
    ValueError
        The ``learning_parameter.loss_function`` must not be an instance of "MaskedLossFunction" when ``mask_dataset=None``.
    """

    # Start counter
    tic = datetime.datetime.now()

    if not isinstance(model, NeuralNetwork):
        raise TypeError(
            f"model must be an instance of NeuralNetwork, not {type(model)}"
        )

    if isinstance(dataset, RegressionDataset):
        pass
    elif (
        isinstance(dataset, Sequence)
        and len(dataset) == 2
        and isinstance(dataset[0], RegressionDataset)
        and isinstance(dataset[1], RegressionDataset)
    ):
        pass
    else:
        raise TypeError(
            f"dataset must be an instance of RegressionDataset or a tuple of two RegressionDataset, not {type(dataset)}"
        )

    if isinstance(mask_dataset, MaskDataset) or mask_dataset is None:
        pass
    elif (
        isinstance(mask_dataset, Sequence)
        and len(mask_dataset) == 2
        and mask_dataset[0] is None
        and mask_dataset[1] is None
    ):
        mask_dataset = None
    elif (
        isinstance(mask_dataset, Sequence)
        and len(mask_dataset) == 2
        and isinstance(mask_dataset[0], MaskDataset)
        and isinstance(mask_dataset[1], MaskDataset)
    ):
        pass
    else:
        raise TypeError(
            f"mask_dataset must be an instance of MaskDataset or a tuple of two MaskDataset or None, not {type(mask_dataset)}"
        )

    if isinstance(dataset, RegressionDataset) and isinstance(mask_dataset, Sequence):
        raise ValueError(
            "mask_dataset must not be a tuple when dataset is a RegressionDataset"
        )
    if isinstance(dataset, Sequence) and isinstance(mask_dataset, MaskDataset):
        raise ValueError(
            "dataset must not be a tuple when mask_dataset is a MaskDataset"
        )

    if not isinstance(learning_parameters, LearningParameters):
        raise TypeError(
            f"learning_parameters must be an instance of LearningParameters, not {type(learning_parameters)}"
        )

    if additional_metrics is None:
        additional_metrics = {}

    if verbose_level is None:
        verbose_level = 0
    elif verbose_level in [0, 1, 2]:
        pass
    elif verbose_level in ["0", "1", "2"]:
        verbose_level = int(verbose_level)
    else:
        verbose_level = 1  # Default

    if max_iter_no_improve is not None:
        assert isinstance(max_iter_no_improve, int)
        assert max_iter_no_improve >= 1

    if seed is not None:
        random.seed(seed)

    if isinstance(dataset, RegressionDataset):
        if train_samples is not None and val_samples is not None:
            pass

        if train_samples is not None and val_samples is None:
            pass

        if train_samples is None and val_samples is None and val_frac is not None:
            n_val = round(val_frac * len(dataset))
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            train_samples, val_samples = indices[n_val:], indices[:n_val]

        if train_samples is None and val_samples is None and val_frac is None:
            train_samples, val_samples = list(range(len(dataset))), None

        if train_samples is not None and val_samples is not None:
            intersect = set(train_samples) & set(val_samples)
            if len(intersect) > 0:
                raise ValueError(
                    "Train dataset and validation dataset must not share samples, here {intersect}"
                )

        train_set = RegressionSubset(dataset, train_samples)
        val_set = (
            RegressionSubset(dataset, val_samples) if val_samples is not None else None
        )

        if mask_dataset is not None:
            train_mask = MaskSubset(mask_dataset, train_samples)
            val_mask = (
                MaskSubset(mask_dataset, val_samples)
                if val_samples is not None
                else None
            )

    else:
        train_set = dataset[0]
        val_set = dataset[1]

        if mask_dataset is not None:
            train_mask = mask_dataset[0]
            val_mask = mask_dataset[1]

    if any(train_set.has_nonfinite()):
        raise ValueError("Non finite values in training dataset")

    # Training loop

    if verbose_level >= 2:
        count = model.count_parameters()
        size, unit = model.count_bytes()
        print("Training initiated")
        print(
            f"{model}: {count:,} learnable parameters ({size:.2f} {unit})", end="\n\n"
        )

    if isinstance(learning_parameters, LearningParameters):
        learning_parameters = [learning_parameters]
    elif not isinstance(learning_parameters, (list, tuple)):
        raise ValueError(
            "learning_parameters must be an instance of LearningParameters or a list of Learning Parameters"
        )
    elif any(not isinstance(p, LearningParameters) for p in learning_parameters):
        raise ValueError(
            "learning_parameters must be an instance of LearningParameters or a list of Learning Parameters"
        )

    train_loss = []
    val_loss = []
    train_metrics = dict.fromkeys(additional_metrics)
    for key in train_metrics:
        train_metrics[key] = []
    val_metrics = dict.fromkeys(additional_metrics)
    for key in val_metrics:
        val_metrics[key] = []
    lr = []
    bs = []

    for learning_parameter in learning_parameters:
        epochs = learning_parameter.epochs
        batch_size = learning_parameter.batch_size
        if batch_size is None:
            batch_size = len(train_set)
        if isinstance(batch_size, BatchScheduler):
            if batch_size.start is None:
                batch_size.start = len(train_set)  # By default, non-stochastic
            if batch_size.stop is None:
                batch_size.stop = len(train_set)  # By default, non-stochastic
        optimizer = learning_parameter.optimizer
        scheduler = learning_parameter.scheduler

        loss_fun = learning_parameter.loss_fun
        if mask_dataset is not None and not isinstance(loss_fun, MaskedLossFunction):
            loss_fun = MaskOverlay(loss_fun)
        if mask_dataset is None and isinstance(loss_fun, MaskedLossFunction):
            raise ValueError(
                "learning_parameter.loss_function must not be an instance of MaskedLossFunction when mask_dataset is None"
            )

        # Dataloaders

        if isinstance(batch_size, BatchScheduler):
            _batch_size = batch_size.get_batch_size()
        else:
            _batch_size = batch_size

        dataloader_train = DataLoader(
            train_set
            if mask_dataset is None
            else TensorDataset(train_set.x, train_set.y, train_mask.m),
            _batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        dataloader_train_eval = DataLoader(
            train_set
            if mask_dataset is None
            else TensorDataset(train_set.x, train_set.y, train_mask.m),
            len(train_set),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        dataloader_val_eval = DataLoader(
            val_set
            if mask_dataset is None
            else TensorDataset(val_set.x, val_set.y, val_mask.m),
            len(val_set),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        n_batchs_train = len(dataloader_train)
        n_batchs_train_eval = 1
        n_batchs_val_eval = 1

        pbar_epoch = tqdm(range(epochs), disable=verbose_level < 1)
        pbar_epoch.set_description("Epoch")

        for epoch in pbar_epoch:
            # Dataloader
            if isinstance(batch_size, BatchScheduler):
                _batch_size = batch_size.get_batch_size()

                dataloader_train = DataLoader(
                    train_set
                    if mask_dataset is None
                    else TensorDataset(train_set.x, train_set.y, train_mask.m),
                    _batch_size,
                    shuffle=True,
                    drop_last=True,
                    pin_memory=True,
                )

            n_batchs_train = len(dataloader_train)

            lr.append(optimizer.param_groups[0]["lr"])
            bs.append(_batch_size)

            # Training
            model.train()

            pbar_batch = tqdm(
                enumerate(dataloader_train),
                leave=False,
                total=n_batchs_train,
                disable=verbose_level < 2,
            )
            pbar_batch.set_description("Batch (training)")
            for _, batch in pbar_batch:
                optimizer.zero_grad(set_to_none=True)

                loss, _ = _batch_processing(
                    model,
                    batch,
                    loss_fun,
                    mask_dataset is None,
                    metrics=additional_metrics,
                )

                loss.backward()
                optimizer.step()

                pbar_batch.set_postfix({"loss": loss.item()})

            model.eval()

            # Evaluation on train set

            sizes = []
            memory_loss = []
            memory_metrics = dict.fromkeys(train_metrics.keys())
            for key in memory_metrics:
                memory_metrics[key] = []

            pbar_batch = tqdm(
                enumerate(dataloader_train_eval),
                leave=False,
                total=n_batchs_train_eval,
                disable=verbose_level < 1,
            )
            pbar_batch.set_description("Batch (model eval)")
            for _, batch in pbar_batch:
                sizes.append(batch[0].size(0))

                with torch.no_grad():
                    loss, mets = _batch_processing(
                        model,
                        batch,
                        loss_fun,
                        mask_dataset is None,
                        metrics=additional_metrics,
                    )

                memory_loss.append(loss.item())
                for key in memory_metrics:
                    memory_metrics[key].append(mets[key].item())

            n_tot = sum(sizes)
            train_loss.append(sum([s / n_tot * l for l, s in zip(memory_loss, sizes)]))
            for key in train_metrics:
                train_metrics[key].append(
                    sum([s / n_tot * val for val, s in zip(memory_metrics[key], sizes)])
                )

            # Evaluation on validation set

            sizes = []
            memory_loss = []
            memory_metrics = dict.fromkeys(val_metrics.keys())
            for key in memory_metrics:
                memory_metrics[key] = []

            pbar_batch = tqdm(
                enumerate(dataloader_val_eval),
                leave=False,
                total=n_batchs_val_eval,
                disable=verbose_level < 2,
            )
            pbar_batch.set_description("Batch (validation)")

            for _, batch in pbar_batch:
                sizes.append(batch[0].size(0))

                with torch.no_grad():
                    loss, mets = _batch_processing(
                        model,
                        batch,
                        loss_fun,
                        mask_dataset is None,
                        metrics=additional_metrics,
                    )

                memory_loss.append(loss.item())
                for key in memory_metrics:
                    memory_metrics[key].append(mets[key].item())

            n_tot = sum(sizes)
            val_loss.append(sum([s / n_tot * l for l, s in zip(memory_loss, sizes)]))
            for key in val_metrics:
                val_metrics[key].append(
                    sum([s / n_tot * val for val, s in zip(memory_metrics[key], sizes)])
                )

            # End of epoch
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_loss[-1])
            else:
                scheduler.step()
            if isinstance(batch_size, BatchScheduler):
                batch_size.step()

            if len(additional_metrics) > 0:
                key = list(additional_metrics.keys())[
                    0
                ]  # Display only the first metric
                add = {
                    f"train {key}": train_metrics[key][-1],
                    f"val  {key}": val_metrics[key][-1],
                }
            else:
                add = {}
            pbar_epoch.set_postfix(
                {
                    "train loss": train_loss[-1],
                    "val loss": val_loss[-1],
                }.update(add)
            )

            # Early stopping
            if (
                max_iter_no_improve is not None
                and epoch > max_iter_no_improve
                and np.min(train_loss[-max_iter_no_improve:]) > np.min(train_loss)
            ):
                break

    model.eval()

    toc = datetime.datetime.now()

    print()

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_set": train_set,
        "val_set": val_set,
        "lr": lr,
        "batch_size": bs,
        "duration": toc - tic,
    }


def _batch_processing(
    model: NeuralNetwork,
    batch: Optional[torch.Tensor],
    loss_fun: Callable,
    masked: bool,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    if masked:
        x, y = batch
        m = None
    else:
        x, y, m = batch

    # Get the data to GPU (if available)
    x = x.to(model.device, non_blocking=True)
    y = y.to(model.device, non_blocking=True)

    if m is not None:
        m = m.to(model.device, non_blocking=True)

    y_hat = model.forward(x)

    if m is None:
        loss = loss_fun(y_hat, y)
    else:
        loss = loss_fun(y_hat, y, m)

    mets = dict.fromkeys(metrics.keys())
    for key in mets:
        if m is None:
            dist = metrics[key](y_hat.detach(), y)
            dist = dist.mean()
        else:
            w = 1.0 / m.sum(dim=0).clip(min=1)
            dist = metrics[key](y_hat.detach(), y)
            dist = (w * dist).sum(dim=0).mean()
        mets[key] = dist

    return loss, mets
