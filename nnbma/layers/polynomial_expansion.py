import itertools as itt
from typing import Union

from numpy import ndarray

import torch
from torch import nn

__all__ = ["PolynomialExpansion"]


class PolynomialExpansion(nn.Module):
    """
    Polynomial expension layer.

    Attributes
    ----------
    att : type
        Description.
    """

    n_features: int
    order: int
    standardize: bool
    device: str
    n_expanded_features: int

    def __init__(
        self,
        n_features: int,
        order: int,
        standardize: bool=True,
        device: str='cpu',
    ):
        """
        Initializer.

        Parameters
        ----------
        param : type
            Description.
        """
        super().__init__()

        self.n_features = n_features
        self.order = order
        self.standardize = standardize
        self.device = device

        # Expanded features
        self.n_expanded_features = PolynomialExpansion.expanded_features(order, n_features)

        # Mask creation
        self._mask = PolynomialExpansion._create_mask(order, n_features)

        # Standardization
        self._batch_norm = nn.BatchNorm1d(self.n_expanded_features) if standardize else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        is1d = x.ndim == 1
        if is1d:
            x = x.unsqueeze(0)
        y = torch.cat((
            torch.ones(x.shape[:-1], device=self.device).unsqueeze(-1),
            x
        ), dim=-1) # Add an input equal to 1
        m = y.clone()
        for _ in range(self.order - 1):
            y = y.unsqueeze(-1)
            m = m.unsqueeze(x.ndim-1) * y
        y = m[..., self._mask]
        if self.standardize:
            y = self._batch_norm(y)
        if is1d:
            y = y.squeeze(0)
        return y

    @staticmethod
    def _create_mask(order: int, n_features: int) -> torch.Tensor:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        mask = torch.ones(order * (n_features + 1,), dtype=bool)
        for coords in itt.product(*(range(n_features + 1) for _ in range(order))):
            # Overlook 0 order expension
            if sum(coords) == 0:
                mask[coords] = False

            # Select only the upper part of the tensor
            for k in range(order - 1):
                if coords[k + 1] < coords[k]:
                    mask[coords] = False
                    break
        return mask

    @staticmethod
    def expanded_features(order: int, n_features: int) -> int:
        """
        Returns the number of augmented polynomial features of order lower or equal to `order` and of `n_features` variables.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        return PolynomialExpansion._create_mask(order, n_features).sum().item()

    def __str__(self) -> str:
        """Returns str(self)"""
        return f"Polynomial expansion ({self.n_features} features, order {self.order})"
