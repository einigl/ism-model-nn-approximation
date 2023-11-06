import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

LOG10 = math.log(10.0)

__all__ = [
    "MaskedLossFunction",
    "MaskOverlay",
    "MaskedMSELoss",
    "CauchyLoss",
    "SmoothL1Loss",
]

## Masked loss functions


class MaskedLossFunction(nn.Module, ABC):
    r"""Implements a masked loss function which has a signature ``loss_fun(y_hat, y, mask)``."""

    def __init__(self):
        r""" """
        super().__init__()

    @abstractmethod
    def forward(
        y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluates the loss between a prediction ``y_hat`` and a reference ``y`` with some masked values indicated in ``mask``.

        Parameters
        ----------
        y_hat : torch.Tensor
            network prediction.
        y : torch.Tensor
            true values. Must have the same shape as ``y_hat``.
        mask : torch.Tensor
            binary mask with values to disregard in the loss. Must have the same shape as ``y_hat``.

        Returns
        -------
        torch.Tensor
            evaluated loss, should be a float.
        """
        pass


class MaskOverlay(MaskedLossFunction):
    r"""Permits to use a MaskedLossFunction as a standard loss function."""

    def __init__(self, loss):
        r"""

        Parameters
        ----------
        loss : Callable
            loss function
        """
        super().__init__()
        self.loss = loss

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, _) -> torch.Tensor:
        """Evaluate the loss function.

        Parameters
        ----------
        y_hat : torch.Tensor
            network prediction.
        y : torch.Tensor
            true values. Must have the same shape as ``y_hat``.

        Returns
        -------
        torch.Tensor
            evaluated loss, should be a float.
        """
        return self.loss(y_hat, y)


class MaskedMSELoss(MaskedLossFunction):
    r"""Implements the masked MSE loss function, i.e., for a binary mask :math:`m`, a prediction :math:`\widehat{y}` and a true value :math:`y`,


    .. math::

       \mathrm{MaskedMSE}(\widehat{y}, y, m) = \frac{1}{\sum_{i} m_{i}} \sum_{i} m_{i} \left( \widehat{y}_{i} - y_{i} \right)^2

    """

    def __init__(self):
        r""" """
        super().__init__()

    def forward(
        self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluates the masked MSE loss between a prediction ``y_hat`` and a reference ``y``, with a binary mask ``m`` -- where :math:`m_{i}=0` corresponds to a masked value.

        Parameters
        ----------
        y_hat : torch.Tensor
            network prediction.
        y : torch.Tensor
            true values. Must have the same shape as ``y_hat``.
        mask : torch.Tensor
            binary mask. Must have the same shape as ``y_hat``.

        Returns
        -------
        torch.Tensor
            evaluated loss, should be a float.
        """
        w = mask / mask.sum(dim=0).clip(min=1)
        return (w * (y_hat - y).square()).sum(dim=0).mean()


# class MaskedWeightedMSELoss(MaskedLossFunction):
#     def __init__(self, w: List):
#         super().__init__()
#         self.w: torch.Tensor = torch.tensor(w).flatten()

#     def forward(
#         self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
#     ) -> torch.Tensor:
#         w = mask / mask.sum(dim=0).clip(min=1)
#         return (self.w * w * (y_hat - y).square()).sum(dim=0).mean()


# class MaskedMAELoss(MaskedLossFunction):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
#         w = mask / mask.sum(dim=0).clip(min=1)
#         return (w * (y_hat - y).abs()).sum(dim=0).mean()


# class MaskedPowerLoss(MaskedLossFunction):
#     degree: int

#     def __init__(self, degree: int):
#         super().__init__()
#         self.degree = degree

#     def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
#         w = mask / mask.sum(dim=0).clip(min=1)
#         return (w * (y_hat - y).abs() ** self.degree).sum(dim=0).mean()


# class MaskedSeriesLoss(MaskedLossFunction):
#     def __init__(self, order: int):
#         super().__init__()
#         self.order = order

#     def forward(
#         self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
#     ) -> torch.Tensor:
#         w = mask / mask.sum(dim=0).clip(min=1)
#         diff = (y_hat - y).abs()
#         err = 0.0 * diff
#         logk = 1.0
#         diffk = 0.0 * diff + 1.0
#         denomk = 1
#         for k in range(1, self.order + 1):
#             logk = logk * LOG10
#             diffk = diffk * diff
#             denomk = denomk * k
#             err = err + logk * diffk / denomk
#         return (w * err).sum(dim=0).mean()


# class MaskedRelErrorLoss(MaskedLossFunction):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
#         w = mask / mask.sum(dim=0).clip(min=1)
#         relerr = (LOG10 * (y_hat - y).abs()).exp() - 1
#         return (w * relerr).sum(dim=0).mean()


# Custom classic loss functions


class CauchyLoss(nn.Module):
    r"""Implements the Cauchy loss function, i.e.,


    .. math::

       \mathrm{CL}(\widehat{y}, y) = \sum_{i} \log \left( 1 + \left( \widehat{y}_{i} - y_{i} \right)^2 \right)

    """

    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Evaluates the Cauchy loss between a prediction ``y_hat`` and a reference ``y``.

        Parameters
        ----------
        y_hat : torch.Tensor
            network prediction.
        y : torch.Tensor
            true values. Must have the same shape as ``y_hat``.

        Returns
        -------
        torch.Tensor
            evaluated loss, should be a float.
        """
        return torch.log(1 + (y_hat - y).square()).mean()


class SmoothL1Loss(nn.Module):
    r"""Implements the smooth L1 loss function, i.e.,


    .. math::

       \mathrm{SmoothL1}(\widehat{y}, y) = \sum_{i} \begin{cases} \frac{1}{2\beta}\left( \widehat{y}_{i} - y_{i} \right)^2 \; \text{ if } \vert \widehat{y}_{i} - y_{i} \vert \leq \beta \\ \vert \widehat{y}_{i} - y_{i} \vert - 0.5 \beta \; \text{ otherwise} \end{cases}

    """
    # beta: float

    def __init__(self, beta: float):
        r"""

        Parameters
        ----------
        beta : float
            :math:`\beta` parameter of the loss function.
        """
        super().__init__()
        self.beta = beta

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Evaluates the smooth L1 loss between a prediction ``y_hat`` and a reference ``y``.

        Parameters
        ----------
        y_hat : torch.Tensor
            network prediction.
        y : torch.Tensor
            true values. Must have the same shape as ``y_hat``.

        Returns
        -------
        torch.Tensor
            evaluated loss, should be a float.
        """
        abs_diffs = torch.abs(y_hat - y)
        return torch.mean(
            torch.where(
                abs_diffs < self.beta,
                0.5 * torch.square(abs_diffs) / self.beta,
                abs_diffs - 0.5 * self.beta,
            )
        )
