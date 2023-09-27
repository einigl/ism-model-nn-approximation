import itertools as itt
from math import comb
from typing import Union

import torch
from numpy import ndarray
from torch import nn

__all__ = ["PolynomialExpansion"]

class PolynomialExpansion(nn.Module):
    """
    Polynomial expansion layer.

    Attributes
    ----------
    att : type
        Description.
    """

    n_features: int
    order: int
    device: str
    n_expanded_features: int

    def __init__(
        self,
        n_features: int,
        order: int,
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
        self.device = device

        # Expanded features
        self.n_expanded_features = PolynomialExpansion.expanded_features(order, n_features)

        # Mask creation
        self._mask = nn.Parameter(
            PolynomialExpansion._create_mask(order, n_features),
            requires_grad=False,
        )

        # Standardization
        self.count = nn.Parameter(torch.tensor(0, dtype=int), requires_grad=False)
        self.means = nn.Parameter(torch.zeros(self.n_expanded_features), requires_grad=False)
        self.squares = nn.Parameter(torch.ones(self.n_expanded_features), requires_grad=False)
        self.stds = nn.Parameter(torch.ones(self.n_expanded_features), requires_grad=False)

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
        y = m.reshape(*x.size()[:-1], -1).matmul(self._mask)
        y = (y - self.means) / self.stds
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
        # Build hypercube
        mask = torch.ones(order * (n_features + 1,), dtype=bool)
        for coords in itt.product(*(range(n_features + 1) for _ in range(order))):

            # Select only the upper part of the tensor
            for k in range(order - 1):
                if coords[k + 1] < coords[k]:
                    mask[coords] = False
                    break

        # Flatten the hypercube
        mask_cube = mask.flatten()

        # Overlook 0 order expansion
        mask_cube[0] = False

        # Build the selection matrix
        n_rows, n_cols = mask_cube.numel(), mask_cube.sum()
        mask_mat = torch.zeros((n_rows, n_cols))
        idx = torch.arange(n_rows)[mask_cube]
        for i in range(n_cols):
            mask_mat[idx[i], i] = 1.

        return mask_mat

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
        return sum([comb(n_features + d - 1, d) for d in range(1, order+1)])

    def update_standardization(
        self,
        x: Union[torch.Tensor, ndarray],
        reset: bool=False
    ) -> None:
        """
        Update the mean and the standard deviation of the polynomial features.
        This operation is not mandatory, but it can be helpful in order to have standardize features at the output of the layer, even in the case of standardized inputs (because polynomial transformations of standardized variables are generally not standardized).
        This fonction can be called multiple times for different batches. To ensure correct moment calculation, no entry must be passed more than once.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        reset: bool, optional
            If True, reset moment calculation.
        """
        if isinstance(x, ndarray):
            x = torch.from_numpy(x)
        prev_mode = self.training
        self.eval()
        y = self.forward(x) * self.stds + self.means
        self.train(mode=prev_mode)

        y = y.reshape(-1, y.size(-1))

        if reset:
            self.count *= 0

        self.count += y.size(0)

        self.means *= 1 - y.size(0)/self.count
        self.means += y.size(0)/self.count * y.mean(dim=0)
        self.squares *= 1 - y.size(0)/self.count
        self.squares += y.size(0)/self.count * y.square().mean(dim=0)

        self.stds *= 0
        self.stds += torch.sqrt(self.squares - self.means**2)

    def __str__(self) -> str:
        """Returns str(self)"""
        return f"Polynomial expansion ({self.n_features} features, order {self.order})"
