import math
from abc import abstractmethod, ABC

from typing import List, Tuple, Optional, Union

import torch
from torch import nn

LOG10 = math.log(10.)

__all__ = [
    "MaskedLossFunction",
    "MaskOverlay",
    "MaskedMSELoss",
    "MaskedWeightedMSELoss",
    "MaskedMAELoss",
    "MaskedPowerLoss",
    "MaskedSeriesLoss",
    "MaskedRelErrorLoss",

    "EvolutiveLossFunction",
    "EvolutiveCoefficients",
    "HierarchicalCoefficients",
    "ProgressiveCoefficients",
    "ProgressivePower",
    "EvolutiveMaskedPowerLoss",
    "EvolutiveMaskedSeriesLoss",
    "EvolutiveMaskedSeriesLossSmooth",

    "CauchyLoss",
    "SmoothL1Loss",
    "QuarticLoss",
    "MixedLoss",
    "MixedLoss2"
]

## Masked loss functions

class MaskedLossFunction(nn.Module, ABC):
    """
    Implements a loss function which has a signature loss_fun(y_hat, y, mask)
    """

    def __init__(self):
        """ Initializer """
        super().__init__()

    @abstractmethod
    def forward(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

class MaskOverlay(MaskedLossFunction):
    """ TODO """

    def __init__(self, loss):
        """ Initializer """
        super().__init__()
        self.loss = loss

    def forward(self, y_hat, y, _):
        return self.loss(y_hat, y)

class MaskedMSELoss(MaskedLossFunction):

    def __init__(self):
        """ Initializer """
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask / mask.sum(dim=0).clip(min=1)
        return (w * (y_hat-y).square()).sum(dim=0).mean()
    
class MaskedWeightedMSELoss(MaskedLossFunction):

    def __init__(self, w: List):
        """ Initializer """
        super().__init__()
        self.w: torch.Tensor = torch.tensor(w).flatten()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask / mask.sum(dim=0).clip(min=1)
        return (self.w * w * (y_hat-y).square()).sum(dim=0).mean()
    
class MaskedMAELoss(MaskedLossFunction):

    def __init__(self):
        """ Initializer """
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        w = mask / mask.sum(dim=0).clip(min=1)
        return (w * (y_hat-y).abs()).sum(dim=0).mean()
    
class MaskedPowerLoss(MaskedLossFunction):

    degree: int

    def __init__(self, degree: int):
        """ Initializer """
        super().__init__()
        self.degree = degree

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        w = mask / mask.sum(dim=0).clip(min=1)
        return (w * (y_hat-y).abs()**self.degree).sum(dim=0).mean()

class MaskedSeriesLoss(MaskedLossFunction):

    def __init__(self, order: int):
        """ Initializer """
        super().__init__()
        self.order = order

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask / mask.sum(dim=0).clip(min=1)
        diff = (y_hat-y).abs()
        err = 0. * diff
        logk = 1.
        diffk = 0. * diff + 1.
        denomk = 1
        for k in range(1, self.order+1):
            logk = logk * LOG10
            diffk = diffk * diff
            denomk = denomk * k
            err = err + logk * diffk / denomk
        return (w * err).sum(dim=0).mean()

class MaskedRelErrorLoss(MaskedLossFunction):

    def __init__(self):
        """ Initializer """
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        w = mask / mask.sum(dim=0).clip(min=1)
        relerr = (LOG10 * (y_hat-y).abs()).exp() - 1
        return (w * relerr).sum(dim=0).mean()
    

## Evolutive loss functions

class EvolutiveLossFunction(nn.Module, ABC):

    epoch: int

    def __init__(self):
        """ Initializer """
        super().__init__()
        self.epoch = 0

    @abstractmethod
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    def step(self) -> None:
        self.epoch += 1
        self.on_epoch_change()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.on_epoch_change()

    @abstractmethod
    def on_epoch_change(self) -> None:
        pass


## Evolutive masked loss functions

class EvolutiveCoefficients(ABC):

    @abstractmethod
    def get(self, epoch: int) -> torch.Tensor:
        pass

class HierarchicalCoefficients(EvolutiveCoefficients):

    def __init__(self, tks: List[Tuple[int, int]]):
        self.tks = tks
        self.n_coeffs = len(self.tks)

    def get(self, epoch: int) -> torch.Tensor:
        a = torch.tensor([tk[0] for tk in self.tks])
        b = torch.tensor([tk[1] for tk in self.tks])
        w = (epoch - a) / (b-a)
        return torch.clip(w, min=0, max=1)
        
class ProgressiveCoefficients(EvolutiveCoefficients):

    def __init__(self, tks: List[int], sks: Union[float, List[float], None]):
        self.tks = tks
        self.n_coeffs = len(self.tks)
        if sks is None:
            self.sks = [1.] * self.n_coeffs
        elif isinstance(sks, (float, int)):
            self.sks = [sks] * self.n_coeffs
        else:
            self.sks = sks
        if len(self.sks) != len(self.tks):
            raise ValueError("tks and sks must have the same number of elements")
        self.sig = torch.nn.Sigmoid()

    def get(self, epoch: int) -> torch.Tensor:
        t0 = torch.tensor(self.tks)
        s = torch.tensor(self.sks)
        return self.sig((epoch-t0)/s)

class ProgressivePower(EvolutiveCoefficients):

    def __init__(self, tks: List[int], sks: Union[float, List[float], None]):
        self.tks = tks
        if sks is None:
            self.sks = [1.] * len(tks)
        elif isinstance(sks, (float, int)):
            self.sks = [sks] * len(tks)
        else:
            self.sks = sks
        if len(self.sks) != len(self.tks):
            raise ValueError("tks and sks must have the same number of elements")
        self.sig = nn.Sigmoid()

    def get(self, epoch: int) -> torch.Tensor:
        return self.sig(torch.tensor(self.tks) - epoch).sum()

class EvolutiveMaskedPowerLoss(EvolutiveLossFunction, MaskedLossFunction):

    coeffs: EvolutiveCoefficients
    w: torch.Tensor

    def __init__(self, pows: ProgressivePower):
        super(EvolutiveLossFunction, self).__init__()
        super(MaskedLossFunction, self).__init__()
        self.pows = pows
        self.p = pows.get(1)
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask / mask.sum(dim=0).clip(min=1)
        return (w * (y_hat-y).abs()**self.p).sum(dim=0).mean()

    def on_epoch_change(self) -> None:
        self.p = self.pows.get(self.epoch)

class EvolutiveMaskedSeriesLoss(EvolutiveLossFunction, MaskedLossFunction):

    coeffs: EvolutiveCoefficients
    w: torch.Tensor

    def __init__(self, coeffs: EvolutiveCoefficients):
        super(EvolutiveLossFunction, self).__init__()
        super(MaskedLossFunction, self).__init__()
        self.coeffs = coeffs
        self.w = coeffs.get(1)
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask / mask.sum(dim=0).clip(min=1)
        diff = (y_hat-y).abs()
        err = 0. * diff
        logk = 1.
        diffk = 0. * diff + 1.
        denomk = 1
        for k in range(1, self.w.numel()+1):
            logk = logk * LOG10
            diffk = diffk * diff
            denomk = denomk * k
            err = err + self.w[k-1] * logk * diffk / denomk
        return (w * err).sum(dim=0).mean()

    def on_epoch_change(self) -> None:
        self.w = self.coeffs.get(self.epoch)

class EvolutiveMaskedSeriesLossSmooth(EvolutiveLossFunction, MaskedLossFunction):

    coeffs: EvolutiveCoefficients
    beta: float
    w: torch.Tensor

    def __init__(self, coeffs: EvolutiveCoefficients, beta: float=1.):
        super(EvolutiveLossFunction, self).__init__()
        super(MaskedLossFunction, self).__init__()
        self.coeffs = coeffs
        self.beta = beta
        self.w = coeffs.get(1)

        self.smoothl1 = nn.SmoothL1Loss(reduction='none', beta=self.beta)
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask / mask.sum(dim=0).clip(min=1)
        diff = self.smoothl1(y_hat, y) 
        err = 0. * diff
        logk = 1.
        diffk = 0. * diff + 1.
        denomk = 1
        for k in range(1, self.w.numel()+1):
            logk = logk * LOG10
            diffk = diffk * diff
            denomk = denomk * k
            err = err + self.w[k-1] * logk * diffk / denomk
        return (w * err).sum(dim=0).mean()

    def on_epoch_change(self) -> None:
        self.w = self.coeffs.get(self.epoch)


## Regular custom loss functions

class CauchyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.log( 1 + (y_hat-y).square() ).mean()

class SmoothL1Loss(nn.Module):

    beta: float

    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        abs_diffs = torch.abs(y_hat - y)
        return torch.mean(
            torch.where(
                abs_diffs < self.beta,
                0.5 * torch.square(abs_diffs) / self.beta,
                abs_diffs - 0.5 * self.beta,
            )
        )

class QuarticLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.square(torch.square(y_hat - y)))

class MixedLoss(nn.Module):

    threshold_in_true: Optional[float]

    def __init__(self, threshold_in_true: Optional[float] = None):
        super().__init__()
        self.threshold_in_true = threshold_in_true

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.where(
                y > self.threshold_in_true,
                torch.square(y_hat - y),
                torch.square(
                    torch.maximum(torch.zeros_like(y), y_hat - self.threshold_in_true)
                ),
            )
        )

class MixedLoss2(nn.Module):

    threshold_in_true: Optional[float] = None,

    def __init__(self, threshold_in_true: Optional[float] = None):
        super().__init__()
        self.threshold_in_true = threshold_in_true

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.where(
                y > self.threshold_in_true,
                torch.square(y_hat - y),
                torch.log(1 + torch.square(y_hat - y)),
            )
        )
