import os
import pickle
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

__all__ = ["MaskDataset", "MaskSubset"]


class MaskDataset(Dataset):
    """Dataset dedicated to ignore some specified values during learning."""

    def __init__(
        self,
        m: np.ndarray,
        features_names: Optional[List[str]] = None,
    ):
        r"""

        Parameters
        ----------
        m : np.ndarray
            Array containing the output features of the regression model.
            y must be of shape :math:`N \times F` where :math:`N` is the number of entries and :math:`F` the number of features.
        features_names : Optional[List[str]], optional
            list of feature names, by default None.

        Raises
        ------
        ValueError
            ``m`` and ``features_names`` must have the same number of features :math:`F`.
        """
        super().__init__()

        self._m: Tensor = from_numpy(m).float()

        if features_names is not None and len(features_names) != m.shape[1]:
            raise ValueError(
                "m and features_names must have the same number of features"
            )

        self._features_names: Optional[List[str]] = features_names

    def __len__(self) -> int:
        r"""
        Returns the number of entries :math:`N` in the dataset.

        Returns
        -------
        int
            Number of entries :math:`N`.
        """
        return self._m.size(0)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        r"""
        Returns the index entries idx.

        Parameters
        ----------
        idx : Any
            Indices of entries to return.

        Returns
        -------
        tuple of torch.Tensor
            Input and output entries.
        """
        return self._m[idx]

    @property
    def m(self) -> Tensor:
        r"""
        Mask Tensor.
        """
        return self._m

    @property
    def n_outputs(self) -> int:
        r"""
        Number of output features.
        """
        return self.m.size(1)

    @property
    def features_names(self) -> Optional[List[str]]:
        r"""
        Features names.
        """
        return self._features_names

    @overload
    def getall(self, numpy: Literal[True]) -> np.ndarray:
        ...

    @overload
    def getall(self, numpy: Literal[False]) -> Tensor:
        ...

    def getall(self, numpy: bool = False) -> Union[np.ndarray, Tensor]:
        r"""
        Returns all the dataset in numpy.ndarray or torch.Tensor depending on the value of the ``numpy`` parameter.

        Parameters
        ----------
        numpy : bool, optional
            If ``numpy==True``, the returned object will be numpy arrays.
            Else, they will be torch tensors.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            Mask.
        """
        m = self[list(range(len(self)))]
        if numpy:
            return m.numpy()
        return m

    @staticmethod
    def from_pandas(df_m: pd.DataFrame) -> "MaskDataset":
        r"""Converts a pandas DataFrame to a MaskDataset object.

        Parameters
        ----------
        df_m : pd.DataFrame
            DataFrame of the masked outputs. This DataFrame should contain :math:`N` rows, i.e., number of entries, and :math:`F` columns, i.e., features.

        Returns
        -------
        MaskDataset
            associated MaskDataset object. The ``m`` attribute is set to values in the DataFrame, and the ``feature_names`` attribute to the column names.
        """
        return MaskDataset(
            df_m.values,
            df_m.columns.to_list(),
        )

    def to_pandas(self) -> pd.DataFrame:
        r"""Converts the mask dataset to two pandas DataFrames.

        Returns
        -------
        pd.DataFrame
            DataFrame of the mask on the output y.
        """
        return pd.DataFrame(self.m, columns=self._features_names)

    def join(self: "MaskDataset", other: "MaskDataset") -> "MaskDataset":
        r"""
        Returns the union of two datasets. Data are copied.

        Parameters
        ----------
        other : MaskDataset
            Other dataset to join with.

        Returns
        -------
        MaskDataset
            New dataset constructed as the union of the two datasets.
        """
        x1, y1 = self.getall(numpy=True)
        x2, y2 = other.getall(numpy=True)
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        return MaskDataset(x, y)

    def substract(self: "MaskDataset", other: "MaskSubset") -> "MaskDataset":
        r"""
        Returns the subtraction of two datasets. Data are copied.

        Parameters
        ----------
        other : MaskDataset
            Subset of ``self``.

        Returns
        -------
        MaskDataset
            New subset of ``self`` containing all values that were not in `other`.
        """
        if not other.issubsetof(self):
            raise ValueError(
                "set2 is not a subset of set1 so it cannot be subtracted."
            )
        new_indices = [i for i in range(len(self)) if i not in other.indices]
        # Algo can be improved.
        return MaskDataset(self, new_indices)

    def stats(self) -> Dict[str, np.ndarray]:
        r"""Computes the proportion of masked entries for each output column.

        Returns
        -------
        Dict[str, np.ndarray]
            dictionary of masked entry proportion for each output feature.
        """
        return {
            "frac": self.m.mean(axis=0).numpy(),
        }

    def save(self, filename: str, path: Optional[str] = None) -> None:
        r"""saves the dataset to a pickle file.

        Parameters
        ----------
        filename : str
            name of the file to be created.
        path : Optional[str], optional
            path to the file to be created, by default None.
        """
        if path is not None:
            filename = os.path.join(path, filename)
        filename = os.path.splitext(filename)[0]
        with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str, path: Optional[str] = None) -> "MaskDataset":
        r"""loads a mask dataset from a pickle file.

        Parameters
        ----------
        filename : str
            name of the file to be read.
        path : Optional[str], optional
            path to the file to be read, by default None.

        Returns
        -------
        MaskDataset
            loaded mask dataset.
        """
        if path is not None:
            filename = os.path.join(path, filename)
        filename = os.path.splitext(filename)[0]
        with open(f"{filename}.pkl", "rb") as file:
            dataset = pickle.load(file)
        return dataset


class MaskSubset(MaskDataset):
    r"""Subset of RegressionDataset."""

    def __init__(self, dataset: MaskDataset, indices: Sequence[int]):
        r"""

        Parameters
        ----------
        dataset : MaskDataset
            Dataset from which entries are extracted.
        indices : Sequence of int
            Indices of entries to extract.
        """
        self._dataset: MaskDataset = dataset
        self._indices: Sequence[int] = indices

    def __len__(self) -> int:
        r"""
        Returns the number of entries in the dataset.

        Returns
        -------
        int
            Number of entries.
        """
        return len(self._indices)

    def __getitem__(self, idx) -> Tensor:
        r"""
        Returns the index entries idx.

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
    def m(self) -> Tensor:
        r"""
        Mask Tensor.
        """
        return self._dataset._m[self._indices]

    def issubsetof(self, dataset: MaskDataset) -> bool:
        r"""
        Returns ``True`` if ``self`` is a subset of ``dataset``.

        Parameters
        ----------
        dataset : MaskDataset
            Dataset of which we want to know if ``self`` is a subset.

        Returns
        -------
        bool
            ``True`` if ``self`` is a subset of ``dataset`` else ``False``.
        """
        # TODO maybe a better solution is suitable
        return dataset == self._dataset
