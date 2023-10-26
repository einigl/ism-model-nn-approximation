from enum import Enum
from typing import Callable, List

import numpy as np

__all__ = ["Operator", "ColumnwiseOperator", "SequentialOperator"]


class Operator:
    r"""Class that stores a transformation to be applied to a full dataset or to one column."""

    def __init__(self, fun: Callable[[np.ndarray], np.ndarray]):
        r"""

        Parameters
        ----------
        fun : Callable[[np.ndarray], np.ndarray]
            any function that returns numpy.arrays with the same size as its input.
        """
        self.fun = fun

    def __call__(self, t: np.ndarray):
        return self.fun(t)

    def __str__(self) -> str:
        return f"Operator: {self.fun.__name__}"


class ColumnwiseOperator(Operator):
    """Class that stores a list of operators -- one per column of the dataset to be considered. It defines a pre-processing on the dataset."""

    def __init__(self, ops: List[Callable[[np.ndarray], np.ndarray]]):
        r"""

        Parameters
        ----------
        ops : List[Callable[[np.ndarray], np.ndarray]]
            list of operators or composition of operators, with one per column of the dataset to be pre-processed.
        """
        self.n_cols = len(ops)
        self.ops = ops

    def __call__(self, t: np.ndarray):
        return np.vstack([self.ops[k](t[:, k]) for k in range(self.n_cols)]).T

    def __str__(self) -> str:
        ops_str = [op.__name__ for op in self.ops]
        return f"ColumnwiseOperator: {ops_str}"


class SequentialOperator(Operator):
    r"""Defines an operator on one column as a composition of multiple operators."""

    def __init__(self, ops: List[Operator]):
        r"""

        Parameters
        ----------
        ops : List[Operator]
            the list of operators to be composed, in the order of application.
        """
        self.ops = ops

    def __call__(self, t: np.ndarray):
        for op in self.ops:
            t = op(t)
        return t

    def __str__(self) -> str:
        ops_str = [str(op) for op in self.ops]
        return f"SequentialOperator: {ops_str}"
