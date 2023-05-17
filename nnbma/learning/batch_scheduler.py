from abc import ABC, abstractmethod

__all__ = [
    "BatchScheduler",
    "LinearBatchScheduler",
    "ExponentialBatchScheduler",
]


class BatchScheduler(ABC):

    epoch: int

    def __init__(self):
        self.epoch = 1

    def step(self) -> None:
        self.epoch += 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @abstractmethod
    def get_batch_size(self) -> int:
        pass

class ConstantBatch():

    batch_size: int

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def get_batch_size(self) -> int:
        return self.get_batch_size

class LinearBatchScheduler(BatchScheduler):

    start: int
    stop: int
    n_epochs: int

    def __init__(self, start: int, stop: int, n_epochs: int):
        super().__init__()
        self.start = start
        self.stop = stop
        self.n_epochs = n_epochs

    def get_batch_size(self) -> int:
        slope = (self.stop - self.start) / self.n_epochs
        return round( (self.epoch - 1) * slope + self.start )
    
class ExponentialBatchScheduler(BatchScheduler):

    start: int
    stop: int
    n_epochs: int

    def __init__(self, start: int, stop: int, n_epochs: int):
        super().__init__()
        self.start = start
        self.stop = stop
        self.n_epochs = n_epochs

    def get_batch_size(self) -> int:
        factor = (self.stop / self.start)**(1/self.n_epochs)
        return round( factor**(self.epoch - 1) * self.start )
