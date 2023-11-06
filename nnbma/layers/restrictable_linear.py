from typing import Optional, Sequence

import torch
from torch import nn

__all__ = ["RestrictableLinear"]


class RestrictableLinear(nn.Linear):
    """
    Restrictable linear layer.
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

        Parameters
        ----------
        in_features : int
            input dimension
        out_features : int
            output dimension
        bias : bool, optional
            use a bias vector, by default True
        device : Optional[str], optional
            device on which the graph should be defined (cpu or cuda), by default None
        dtype : _type_, optional
            dtype to be used in computations, by default None
        outputs_names : Optional[Sequence[str]], optional
            sequence of output names, by default None
        """
        super().__init__(in_features, out_features, bias, device, dtype)

        self.outputs_names = outputs_names

        self.subweight = None
        self.subbias = None

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the linear layer, restricted or not depending of past settings.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (?, ``in_features``)

        Returns
        -------
        torch.Tensor
            output tensor. This tensor has shape (?, ``out_features``) if the full output set is considered, and with less outputs features in case of restricted output.
        """
        if self.training:
            return nn.functional.linear(x, self.weight, self.bias)
        return nn.functional.linear(x, self.subweight, self.subbias)

    def restrict_to_output_subset(self, indices: Sequence[int]) -> None:
        """Restricts the output to an output subset.

        Parameters
        ----------
        indices : Sequence[int]
            index subset to predict.

        Raises
        ------
        PermissionError
            the restriction cannot be applied in train mode.
            To apply the restriction, first turn the model to ``.eval()`` mode.
        """
        if self.training:
            raise PermissionError(
                "You're not able to restrict the outputs when Module mode is train"
            )
        self.subweight = self.weight.data[indices, :]
        if self.bias is not None:
            self.subbias = self.bias.data[indices]

    def train(self, mode: bool = True) -> "RestrictableLinear":
        r"""Sets the object in train mode if ``mode is True`` and in eval mode else.

        Parameters
        ----------
        mode : bool, optional
            Whether the layer is to be set in train mode (``True``) or eval mode (``False``), by default ``True``.

        Returns
        -------
        RestrictableLinear
            The updated object.
        """
        super().train(mode)
        if mode:
            self.subweight = None
            self.subbias = None
        else:
            self.subweight = self.weight.data
            self.subbias = None if self.bias is None else self.bias.data

    def __str__(self) -> str:
        return f"Restrictable layer ({self.in_features} input features, {self.out_features} output features)"
