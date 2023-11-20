from math import log

import torch

__all__ = [
    "error_factor_log",
    "error_factor_lin",
    "relative_error",
    "squared_error",
    "absolute_error",
]

LOG10 = log(10)


def error_factor_log(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    err = torch.exp(LOG10 * torch.abs(y_hat.detach() - y))
    return 100 * (err - 1)


def error_factor_lin(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    err = torch.maximum(torch.abs(y_hat / y), torch.abs(y / y_hat))
    return 100 * (err - 1)


def relative_error(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 100 * (y_hat - y) / y


def squared_error(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y_hat - y).square()


def absolute_error(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y_hat - y).abs()
