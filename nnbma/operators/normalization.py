from enum import Enum

import pandas as pd

from .operator import Operator

__all__ = ["NormTypes", "Normalizer", "InverseNormalizer"]


class NormTypes(Enum):
    """Types of normalization"""

    NONE = 0
    """No normalization."""

    MEAN0 = 1
    """Center the columns, i.e., set their means to 0."""

    STD1 = 2
    """Reduce the columns, i.e., set their variances to 1."""

    MEAN0STD1 = 3
    """Center and reduce the columns, i.e., set their means to 0 and their variances to 1."""

    MIN0MAX1 = 4
    """Apply a MinMax normalization, i.e., set the minimum value of each column to 0 and the maximum to 1."""

    MIN1MAX1 = 5
    """Apply an alternative MinMax normalization, i.e., set the minimum value of each column to -1 and the maximum to 1."""


class Normalizer(Operator):
    r"""Specific operator that applies a specified normalization of the dataset."""

    def __init__(self, df: pd.DataFrame, norm_type: NormTypes):
        """

        Parameters
        ----------
        df : pd.DataFrame
            dataset to be normalized.
        norm_type : NormTypes
            type of normalization to be applied.
        """
        self.norm_type = norm_type

        self.stats = {
            "mean": df.mean().values,
            "std": df.std().values,
            "min": df.min().values,
            "max": df.max().values,
        }

    def __call__(self, t):
        """applies the specified normalization type to a set of values ``t``.

        Parameters
        ----------
        t : numpy.ndarray or pd.DataFrame
            set of values to normalize (typically one column of a dataset).

        Returns
        -------
        numpy.ndarray or pd.DataFrame
            normalized set of values.

        Raises
        ------
        RuntimeError
            The ``norm_type`` attribute is not a valid element of NormTypes.
        """
        if self.norm_type is NormTypes.NONE:
            return t
        elif self.norm_type is NormTypes.MEAN0:
            return t - self.stats["mean"]
        elif self.norm_type is NormTypes.STD1:
            return t / self.stats["std"]
        elif self.norm_type is NormTypes.MEAN0STD1:
            return (t - self.stats["mean"]) / self.stats["std"]
        elif self.norm_type is NormTypes.MIN0MAX1:
            return (t - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
        elif self.norm_type is NormTypes.MIN1MAX1:
            return (
                2 * (t - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
                - 1
            )
        else:
            raise RuntimeError("Should never been here")

    def __str__(self) -> str:
        return f"Normalizer: {self.norm_type}"


class InverseNormalizer(Operator):
    r"""Specific operator that reverts a specified normalization of the dataset."""

    def __init__(self, df: pd.DataFrame, norm_type: NormTypes):
        """

        Parameters
        ----------
        df : pd.DataFrame
            dataset to be normalized.
        norm_type : NormTypes
            type of normalization to revert.
        """
        self.norm_type = norm_type

        self.stats = {
            "mean": df.mean().values,
            "std": df.std().values,
            "min": df.min().values,
            "max": df.max().values,
        }

    def __call__(self, t):
        """reverts the specified normalization type to a set of normalized values ``t``.

        Parameters
        ----------
        t : numpy.ndarray or pd.DataFrame
            normalized set of values (typically one column of a dataset).

        Returns
        -------
        numpy.ndarray or pd.DataFrame
            unnormalized set of values

        Raises
        ------
        RuntimeError
            The ``norm_type`` attribute is not a valid element of NormTypes.
        """
        if self.norm_type is NormTypes.NONE:
            return t
        elif self.norm_type is NormTypes.MEAN0:
            return t + self.stats["mean"]
        elif self.norm_type is NormTypes.STD1:
            return t * self.stats["std"]
        elif self.norm_type is NormTypes.MEAN0STD1:
            return t * self.stats["std"] + self.stats["mean"]
        elif self.norm_type is NormTypes.MIN0MAX1:
            return t * (self.stats["max"] - self.stats["min"]) + self.stats["min"]
        elif self.norm_type is NormTypes.MIN1MAX1:
            return (t + 1) * (self.stats["max"] - self.stats["min"]) / 2 + self.stats[
                "min"
            ]
        else:
            raise RuntimeError("Should never been here")

    def __str__(self) -> str:
        return f"Inverse normalizer: {self.norm_type}"
