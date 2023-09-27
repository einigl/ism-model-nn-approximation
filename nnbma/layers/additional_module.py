from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

__all__ = ["AdditionalModule"]

class AdditionalModule(nn.Module, ABC):
    """
    Additional module.

    Attributes
    ----------
    input_features: int
        Number of input features.
    output_features: int
        Number of output features.
    device: str
        Device to use. Default: device.
    """

    input_features: Optional[int]
    output_features: Optional[int]
    device: str

    def __init__(
        self,
        input_features: Optional[int],
        output_features: Optional[int],
        device: str='cpu',
    ):
        """
        Initializer.

        Parameters
        ----------
        input_features: int
            Number of input features.
        output_features: int
            Number of output features.
        device: str
            Device to use. Default: device
        """
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """

    def __str__(self) -> str:
        """
        Returns str(self)
        """
        return f"Additional module ({self.input_features} input features, {self.output_features} output features)"
