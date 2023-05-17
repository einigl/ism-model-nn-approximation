from typing import Optional, Sequence

import torch
from torch import nn

__all__ = ["RestrictableLinear"]


class RestrictableLinear(nn.Linear):
    """
    Restrictable linear layer.

    Attributes
    ----------
    att : type
        Description.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype=None,
        outputs_names: Optional[Sequence[str]] = None,
    ):
        """
        Initializer.

        Parameters
        ----------
        param : type
            Description.
        """
        super().__init__(in_features, out_features, bias, device, dtype)

        self.outputs_names = outputs_names

        self.subweight = None
        self.subbias = None

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
        if self.training:
            return nn.functional.linear(x, self.weight, self.bias)
        return nn.functional.linear(x, self.subweight, self.subbias)

    def restrict_to_output_subset(self, indices: Sequence[int]) -> None:
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
        if self.training:
            raise PermissionError(
                "You're not able to restrict the outputs when Module mode is train"
            )
        self.subweight = self.weight.data[indices, :]
        if self.bias is not None:
            self.subbias = self.bias.data[indices]

    def train(self, mode: bool = True) -> "RestrictableLinear":
        super().train(mode)
        if mode:
            self.subweight = None
            self.subbias = None
        else:
            self.subweight = self.weight.data
            self.subbias = None if self.bias is None else self.bias.data

    def __str__(self) -> str:
        """ Returns str(self) """
        return f"Restrictable layer ({self.in_features} input features, {self.out_features} output features)"
