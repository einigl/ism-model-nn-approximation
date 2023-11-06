from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import torch
from torch import nn

__all__ = ["AdditionalModule", "AdditionalModuleFromExisting"]


class AdditionalModule(nn.Module, ABC):
    r"""
    Additional module.
    """

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


class AdditionalModuleFromExisting(AdditionalModule):
    r"""
    Additional module build from an existing Torch module or function.
    Avoid overriding forward method.
    """

    def __init__(
        self,
        input_features: Optional[int],
        output_features: Optional[int],
        operation: Union[nn.Module, Callable],
        device: str = "cpu",
    ):
        r"""

        Parameters
        ----------
        input_features: int
            Number of input features.
        output_features: int
            Number of output features.
        operation: Module | function
            Operation to apply (Torch module or function).
        device: str
            Device to use, by default "cpu".
        """
        super().__init__(input_features, output_features, device=device)

        self.operation = operation

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
        if isinstance(self.operation, nn.Module):
            return self.operation.forward(
                x
            )  # To prevent cases where __call__ is overriden and may be different from forward
        return self.operation(x)

    def __str__(self) -> str:
        return f"Additional module from an existing Torch operator ({self.input_features} input features, {self.output_features} output features, module: {self.module})"
