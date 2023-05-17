import os
import pickle
from typing import Callable, List, Dict, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

__all__ = ["RegressionDataset", "RegressionSubset"]


class RegressionDataset(Dataset):
    """Dataset dedicated to regression."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        inputs_names: Optional[List[str]] = None,
        outputs_names: Optional[List[str]] = None,
    ):
        """
        Initializer.

        Parameters
        ----------
        x : numpy.ndarray
            Array containing the input features of the regression model.
            x must be of shape N x I where N is the number of entries and I the number of input features.
        y : numpy.ndarray
            Array containing the output features of the regression model.
            y must be of shape N x O where N is the number of entries and O the number of output features.
        """
        super().__init__()

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows.")

        self._x: Tensor = from_numpy(x).float()
        self._y: Tensor = from_numpy(y).float()

        if inputs_names is not None and len(inputs_names) != x.shape[1]:
            raise ValueError("x and inputs_names must have the same number of features")
        if outputs_names is not None and len(outputs_names) != y.shape[1]:
            raise ValueError(
                "y and outputs_names must have the same number of features"
            )

        self._inputs_names: Optional[List[str]] = inputs_names
        self._outputs_names: Optional[List[str]] = outputs_names

    def __len__(self) -> int:
        """
        Returns the number of entries in the dataset.

        Returns
        -------
        int
            Number of entries.
        """
        return self._x.size(0)
    
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        Returns the entries of indice(s) idx.

        Parameters
        ----------
        idx : Any
            Indices of entries to return.

        Returns
        -------
        tuple of torch.Tensor
            Input and output entries.
        """
        return self._x[idx], self._y[idx]
    
    @property
    def x(self) -> Tensor:
        """
        Input tensor.
        """
        return self._x
    
    @property
    def y(self) -> Tensor:
        """
        Output tensor.
        """
        return self._y

    @property
    def n_inputs(self) -> int:
        """
        Number of input features.
        """
        return self.x.size(1)

    @property
    def n_outputs(self) -> int:
        """
        Number of output features.
        """
        return self.y.size(1)

    @property
    def inputs_names(self) -> int:
        """
        Inputs names.
        """
        return self._inputs_names

    @property
    def outputs_names(self) -> int:
        """
        Outputs names.
        """
        return self._outputs_names

    def has_nan(self) -> Tuple[bool, bool]:
        """
        Returns a tuple of two boolean.
        The first one is True if the input features contain at least one NaN, else False.
        The second one is True if the output features contain at least one NaN, else False.

        Returns
        -------
        tuple of bool
            Evaluate the presence of NaN in the input and output sets.
        """
        return self.x.isnan().any().item(), self.y.isnan().any().item()

    def has_nonfinite(self) -> Tuple[bool, bool]:
        """
        Returns a tuple of two boolean.
        The first one is True if the input features contain at least one non finite value (including NaNs), else False.
        The second one is True if the output features contain at least one non finite value (including NaNs), else False.

        Returns
        -------
        tuple of bool
            Evaluate the presence of non finite values in the input and output sets.
        """
        return not self.x.isfinite().all().item(), not self.y.isfinite().all().item()

    def apply_transf(
        self,
        x_op: Optional[Callable[[np.ndarray], np.ndarray]],
        y_op: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> "RegressionDataset":
        """
        Apply an operator to x and y. A new dataset is returned so the operators should not use in-place operations.

        Parameters
        ----------
        x_op : Callable[[numpy.ndarray], np.ndarray]
            Operator to apply on the input features.
        y_op : Callable[[numpy.ndarray], np.ndarray]
            Operator to apply on the output features.

        Returns
        -------
        RegressionDataset
            New dataset with transformed values.
        """
        if x_op is None:
            x_op = lambda t: t
        if y_op is None:
            y_op = lambda t: t
        return type(self)(
            x_op(self.x.numpy()),
            y_op(self.y.numpy()),
            self._inputs_names,
            self._outputs_names,
        )

    @overload
    def getall(self, numpy: Literal[True]) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def getall(self, numpy: Literal[False]) -> Tuple[Tensor, Tensor]:
        ...

    def getall(self, numpy: bool = False) -> Tuple[Union[np.ndarray, Tensor], ...]:
        """
        Returns all the dataset in numpy.ndarray or torch.Tensor depending on the value of the `numpy` parameter.

        Parameters
        ----------
        numpy : bool, optional
            If `numpy` is True, the returned object will be numpy arrays.
            Else, they will be torch tensors.

        Returns
        -------
        tuple of torch.Tensor or numpy.ndarray
            Inputs and outputs sets.
        """
        x, y = self[list(range(len(self)))]
        if numpy:
            return x.numpy(), y.numpy()
        return x, y

    @staticmethod
    def from_pandas(df_x: pd.DataFrame, df_y: pd.DataFrame) -> "RegressionDataset":
        return RegressionDataset(
            df_x.values,
            df_y.values,
            df_x.columns.to_list(),
            df_y.columns.to_list(),
        )

    def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(self.x, columns=self._inputs_names), pd.DataFrame(
            self.y, columns=self._outputs_names
        )

    def join(
        self: "RegressionDataset", other: "RegressionDataset"
    ) -> "RegressionDataset":
        """
        Returns the union of two datasets. Data are copied.

        Parameters
        ----------
        other : RegressionDataset
            Other dataset to join with.

        Returns
        -------
        type
            New dataset constructed as the union of the two datasets.
        """
        x1, y1 = self.getall(numpy=True)
        x2, y2 = other.getall(numpy=True)
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        return RegressionDataset(x, y)

    def substract(
        self: "RegressionDataset", other: "RegressionSubset"
    ) -> "RegressionSubset":
        """
        Returns the substraction of two datasets. Data are copied.

        Description.

        Parameters
        ----------
        other : RegressionSubset
            Subset of `self`.

        Returns
        -------

        RegressionSubset
            New subset of `self` containing all values that were not in `other`.
        """
        if not other.issubsetof(self):
            raise ValueError(
                "set2 is not a subset of set1 so it cannot be substracted."
            )
        new_indices = [i for i in range(len(self)) if i not in other._indices]
        # Algo can be improved.
        return RegressionSubset(self, new_indices)

    def stats(self) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray]
    ]:
        return {
            "mean": self.x.mean(axis=0).numpy(),
            "std": self.x.std(axis=0).numpy(),
            "min": self.x.min(axis=0).numpy(),
            "max": self.x.max(axis=0).numpy(),
        }, {
            "mean": self.y.mean(axis=0).numpy(),
            "std": self.y.std(axis=0).numpy(),
            "min": self.y.min(axis=0).numpy(),
            "max": self.y.max(axis=0).numpy(),
        }

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """TODO"""
        if path is not None:
            filename = os.path.join(path, filename)
        filename = os.path.splitext(filename)[0]
        with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str, path: Optional[str] = None) -> "RegressionDataset":
        """TODO"""
        if path is not None:
            filename = os.path.join(path, filename)
        filename = os.path.splitext(filename)[0]
        with open(f"{filename}.pkl", "rb") as file:
            dataset = pickle.load(file)
        return dataset


class RegressionSubset(RegressionDataset):
    """Subset of RegressionDataset."""

    def __init__(self, dataset: RegressionDataset, indices: Sequence[int]):
        """
        Initializer.

        Parameters
        ----------
        dataset : RegressionDataset
            Dataset from which entries are extracted.
        indices : Sequence of int
            Indices of entries to extract.
        """
        self._dataset: RegressionDataset = dataset
        self._indices: Sequence[int] = indices

    def __len__(self) -> int:
        """
        Returns the number of entries in the dataset.

        Returns
        -------
        int
            Number of entries.
        """
        return len(self._indices)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        Returns the entries of indice(s) idx.

        Parameters
        ----------
        idx : Any
            Indices of entries to return.

        Returns
        -------
        tuple of torch.Tensor
            Input and output entries.
        """
        return self._dataset[self._indices[idx]]

    @property
    def x(self) -> Tensor:
        """
        Input tensor.
        """
        return self._dataset._x[self._indices]
    
    @property
    def y(self) -> Tensor:
        """
        Output tensor.
        """
        return self._dataset._y[self._indices]

    def issubsetof(self, dataset: RegressionDataset) -> bool:
        """
        Returns True of `self` is a subset of `dataset`.

        Parameters
        ----------
        dataset : RegressionDataset
            Dataset of which we want to know if `self` is a subset.

        Returns
        -------
        bool
            True of `self` is a subset of `dataset` else False.
        """
        # TODO maybe a better solution is suitable
        return dataset == self._dataset
