from abc import ABC, abstractmethod

__all__ = [
    "BatchScheduler",
    "LinearBatchScheduler",
    "ExponentialBatchScheduler",
]


class BatchScheduler(ABC):
    r"""Abstract class for schedulers of the batch size :math:`b(t)` with respect to the ``epoch`` attribute, denoted :math:`t` in math equations."""

    def __init__(self):
        self.epoch = 1

    def step(self) -> None:
        r"""Increments the ``epoch`` attribute: :math:`t \leftarrow t+1`."""
        self.epoch += 1

    def set_epoch(self, epoch: int) -> None:
        """Set the ``epoch`` attribute to a new value.

        Parameters
        ----------
        epoch : int
            new value for the ``epoch`` attribute.
        """
        self.epoch = epoch

    @abstractmethod
    def get_batch_size(self) -> int:
        """Returns the batch size :math:`b(t)` for the next epoch depending on the number of already run epochs :math:`t`.

        Returns
        -------
        int
            batch size :math:`b(t)`.
        """
        pass


class ConstantBatch:
    r"""Simplest scheduler, that always returns a predefined constant :math:`c`, i.e., :math:`b(t) = c` for all :math:`t`"""
    # batch_size: int

    def __init__(self, batch_size: int):
        """

        Parameters
        ----------
        batch_size : int
            batch size to consider during the training.
        """
        super().__init__()
        self.batch_size = batch_size

    def get_batch_size(self) -> int:
        return self.get_batch_size


class LinearBatchScheduler(BatchScheduler):
    r"""Scheduler based on a linear interpolation between an initial batch size :math:`b_{i}` and a final batch size :math:`b_{f}` for a total number of epochs :math:`t_{f}`, i.e.,  for :math:`0 \leq t \leq t_{f}`,


    .. math::

       b(t) = \frac{t}{t_{f}} b_{i} + \frac{t_{f} - t}{t_{f}} b_{f}

    """

    # start: int
    # stop: int
    # n_epochs: int

    def __init__(self, start: int, stop: int, n_epochs: int):
        r"""

        Parameters
        ----------
        start : int
            starting value for the batch size :math:`b_{i}`.
        stop : int
            final value for the batch size :math:`b_{f}`.
        n_epochs : int
            total number of epochs considered for training :math:`t_{f}`.
        """
        super().__init__()
        self.start = start
        self.stop = stop
        self.n_epochs = n_epochs

    def get_batch_size(self) -> int:
        slope = (self.stop - self.start) / self.n_epochs
        return round((self.epoch - 1) * slope + self.start)


class ExponentialBatchScheduler(BatchScheduler):
    r"""Scheduler based on an exponential interpolation between an initial batch size :math:`b_{i}` and a final batch size :math:`b_{f}` for a total number of epochs :math:`t_{f}`, i.e., for :math:`0 \leq t \leq t_{f}`,


    .. math::

       b(t) = \left( \frac{b_{f}}{b_{i}} \right)^{t / t_{f}} b_{i}

    """
    # start: int
    # stop: int
    # n_epochs: int

    def __init__(self, start: int, stop: int, n_epochs: int):
        r"""

        Parameters
        ----------
        start : int
            starting value for the batch size :math:`b_{i}`.
        stop : int
            final value for the batch size :math:`b_{f}`.
        n_epochs : int
            total number of epochs considered for training :math:`t_{f}`.
        """
        super().__init__()
        self.start = start
        self.stop = stop
        self.n_epochs = n_epochs

    def get_batch_size(self) -> int:
        factor = (self.stop / self.start) ** (1 / self.n_epochs)
        return round(factor ** (self.epoch - 1) * self.start)
