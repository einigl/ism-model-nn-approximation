from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

__all__ = ["AdditionalModule"]


class AdditionalModule(nn.Module, ABC):
    r"""
    Additional module.
    """

    # list of attributes:
    # input_features: Optional[int]
    # output_features: Optional[int]
    # device: str

    def __init__(
        self,
        input_features: Optional[int],
        output_features: Optional[int],
        device: str = "cpu",
    ):
        r"""

        Parameters
        ----------
        input_features: int
            Number of input features.
        output_features: int
            Number of output features.
        device: str
            Device to use, by default "cpu".
        """
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluates the associated pytorch function.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (?, ``input_features``).

        Returns
        -------
        Tensor
            Output tensor of shape (?, ``output_features``).
        """

    def __str__(self) -> str:
        return f"Additional module ({self.input_features} input features, {self.output_features} output features)"
