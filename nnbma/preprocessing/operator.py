from enum import Enum
from typing import List, Callable

import numpy as np

__all__ = ['Operator', 'ColumnwiseOperator', 'SequentialOperator']

class Operator:
    """TODO"""

    def __init__(self, fun: Callable[[np.ndarray], np.ndarray]):
        self.fun = fun

    def __call__(self, t: np.ndarray):
        return self.fun(t)

    def __str__(self) -> str:
        return f"Operator: {self.fun.__name__}"

class ColumnwiseOperator(Operator):
    """TODO"""

    def __init__(self, ops: List[Callable[[np.ndarray], np.ndarray]]):
        self.n_cols = len(ops)
        self.ops = ops

    def __call__(self, t: np.ndarray):
        return np.vstack([self.ops[k](t[:, k]) for k in range(self.n_cols)]).T

    def __str__(self) -> str:
        ops_str = [op.__name__ for op in self.ops]
        return f"ColumnwiseOperator: {ops_str}"

class SequentialOperator(Operator):
    """TODO"""

    def __init__(self, ops: List[Operator]):
        self.ops = ops

    def __call__(self, t: np.ndarray):
        for op in self.ops:
            t = op(t)
        return t

    def __str__(self) -> str:
        ops_str = [str(op) for op in self.ops]
        return f"SequentialOperator: {ops_str}"
