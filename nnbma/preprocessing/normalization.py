from enum import Enum

import pandas as pd

from .operator import Operator

__all__ = ['NormTypes', 'Normalizer', 'InverseNormalizer']

class NormTypes(Enum):
    """Types of normalization"""

    NONE = 0
    MEAN0 = 1
    STD1 = 2
    MEAN0STD1 = 3
    MIN0MAX1 = 4
    MIN1MAX1 = 5

class Normalizer(Operator):
    """ TODO """

    def __init__(self,
        df: pd.DataFrame,
        norm_type: NormTypes
    ):
        """ TODO """
        self.norm_type = norm_type

        self.stats = {
            "mean": df.mean().values,
            "std": df.std().values,
            "min": df.min().values,
            "max": df.max().values,
        }

    def __call__(self, t):
        """ TODO """
        if self.norm_type is NormTypes.NONE:
            return t
        elif self.norm_type is NormTypes.MEAN0:
            return t - self.stats["mean"]
        elif self.norm_type is NormTypes.STD1:
            return t / self.stats["std"]
        elif self.norm_type is NormTypes.MEAN0STD1:
            return (t - self.stats["mean"]) / self.stats["std"]
        elif self.norm_type is NormTypes.MIN0MAX1:
            return (t-self.stats["min"]) / (self.stats["max"]-self.stats["min"])
        elif self.norm_type is NormTypes.MIN1MAX1:
            return 2 * (t-self.stats["min"]) / (self.stats["max"]-self.stats["min"]) - 1
        else:
            raise RuntimeError('Should never been here')

    def __str__(self) -> str:
        """ Returns str(self) """
        return f"Normalizer: {self.norm_type}"
        
class InverseNormalizer(Operator):
    """ TODO """

    def __init__(self,
        df: pd.DataFrame,
        norm_type: NormTypes
    ):
        """ TODO """
        self.norm_type = norm_type

        self.stats = {
            "mean": df.mean().values,
            "std": df.std().values,
            "min": df.min().values,
            "max": df.max().values,
        }

    def __call__(self, t):
        """ TODO """
        if self.norm_type is NormTypes.NONE:
            return t
        elif self.norm_type is NormTypes.MEAN0:
            return t + self.stats["mean"]
        elif self.norm_type is NormTypes.STD1:
            return t * self.stats["std"]
        elif self.norm_type is NormTypes.MEAN0STD1:
            return t * self.stats["std"] + self.stats["mean"]
        elif self.norm_type is NormTypes.MIN0MAX1:
            return t * (self.stats["max"]-self.stats["min"]) + self.stats["min"]
        elif self.norm_type is NormTypes.MIN1MAX1:
            return (t+1) * (self.stats["max"]-self.stats["min"]) / 2 + self.stats["min"]
        else:
            raise RuntimeError('Should never been here')

    def __str__(self) -> str:
        """ Returns str(self) """
        return f"Inverse normalizer: {self.norm_type}"
