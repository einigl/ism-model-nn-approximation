from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Union, overload

import numpy as np
import torch
from torch import Tensor, nn


class Function(nn.Module, ABC):
    """
    Abstract vectorial function.

    The code of the function is contained in the `forward` method.
    The `__call__` method is a convenience method that allows to call this function in a similar way to any Python functions.
    If the input is a NumPy `ndarray`, the output will be a `ndarray` and if the input is a PyTorch `Tensor`, the output will also be a `Tensor`.
    """

    @abstractmethod
    def forward(self, t: Tensor) -> Tensor:
        pass

    @overload
    def __call__(self, x: Tensor) -> Tensor:
        ...

    @overload
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            return self.forward(torch.from_numpy(x)).numpy()
        return self.forward(x)

    @staticmethod
    def latex() -> str:
        """LaTeX representation of the function"""
        return "Not defined"


class Fexample(Function):
    """
    Implements the (t1, t2) -> (t1+2*t2, t1^2, t1*t2^2) function as a Torch Module.
    """

    n_inputs: int = 2
    n_outputs: int = 3

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 0.0])
    c = torch.tensor([1, 2])

    def forward(self, t: Tensor) -> Tensor:
        """
        This function takes batched inputs, so the inputs must have a shape [N, 2] where N is the batch size i.e. the number of inputs that are computed simultaneously.
        The output is of shape [N, 3].
        """
        if t.shape[-1] != self.n_inputs:
            raise ValueError(f"t.shape[-1] must be {self.n_inputs}, not {t.shape[1]}")
        return torch.concat(
            (
                (self.a * t).sum(-1, keepdim=True),
                (self.b * t**2).sum(-1, keepdim=True),
                (t**self.c).prod(-1, keepdim=True),
            ),
            dim=-1,
        )

    def latex() -> str:
        return r"$$\left(\begin{array}{c} t_1\\ t_2 \end{array}\right) \longmapsto \left(\begin{array}{c} t_1+2t_2\\ t_1^2\\ t_1t_2^2 \end{array}\right)$$"
