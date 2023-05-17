import os
import pickle
from typing import Callable, List, Dict, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

__all__ = ["MaskDataset", "MaskSubset"]


class MaskDataset(Dataset):
    """Dataset dedicated to ignore values during learning."""

    def __init__(
        self,
        m: np.ndarray,
        features_names: Optional[List[str]] = None,
    ):
        """
        Initializer.

        Parameters
        ----------
        m : numpy.ndarray
            Array containing the output features of the regression model.
            y must be of shape N x F where N is the number of entries and F the number of features.
        """
        super().__init__()

        self._m: Tensor = from_numpy(m).float()

        if features_names is not None and len(features_names) != m.shape[1]:
            raise ValueError(
                "m and outputs_names must have the same number of features"
            )

        self._features_names: Optional[List[str]] = features_names

    def __len__(self) -> int:
        """
        Returns the number of entries in the dataset.

        Returns
        -------
        int
            Number of entries.
        """
        return self._m.size(0)
    
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
        return self._m[idx]
    
    @property
    def m(self) -> Tensor:
        """
        Mask Tensor.
        """
        return self._m

    @property
    def n_outputs(self) -> int:
        """
        Number of output features.
        """
        return self.m.size(1)

    @property
    def features_names(self) -> int:
        """
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
        """
        Returns all the dataset in numpy.ndarray or torch.Tensor depending on the value of the `numpy` parameter.

        Parameters
        ----------
        numpy : bool, optional
            If `numpy` is True, the returned object will be numpy arrays.
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
        return MaskDataset(
            df_m.values,
            df_m.columns.to_list(),
        )

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.m, columns=self._features_names)

    def join(
        self: "MaskDataset", other: "MaskDataset"
    ) -> "MaskDataset":
        """
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

    def substract(
        self: "MaskDataset", other: "MaskSubset"
    ) -> "MaskDataset":
        """
        Returns the substraction of two datasets. Data are copied.

        Description.

        Parameters
        ----------
        other : MaskDataset
            Subset of `self`.

        Returns
        -------

        MaskDataset
            New subset of `self` containing all values that were not in `other`.
        """
        if not other.issubsetof(self):
            raise ValueError(
                "set2 is not a subset of set1 so it cannot be substracted."
            )
        new_indices = [i for i in range(len(self)) if i not in other.indices]
        # Algo can be improved.
        return MaskDataset(self, new_indices)

    def stats(self) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray]
    ]:
        return {
            "frac": self.m.mean(axis=0).numpy(),
        }

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """TODO"""
        if path is not None:
            filename = os.path.join(path, filename)
        filename = os.path.splitext(filename)[0]
        with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str, path: Optional[str] = None) -> "MaskDataset":
        """TODO"""
        if path is not None:
            filename = os.path.join(path, filename)
        filename = os.path.splitext(filename)[0]
        with open(f"{filename}.pkl", "rb") as file:
            dataset = pickle.load(file)
        return dataset


class MaskSubset(MaskDataset):
    """Subset of RegressionDataset."""

    def __init__(self, dataset: MaskDataset, indices: Sequence[int]):
        """
        Initializer.

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
        """
        Returns the number of entries in the dataset.

        Returns
        -------
        int
            Number of entries.
        """
        return len(self._indices)

    def __getitem__(self, idx) -> Tensor:
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
    def m(self) -> Tensor:
        """
        Mask Tensor.
        """
        return self._dataset._m[self._indices]

    def issubsetof(self, dataset: MaskDataset) -> bool:
        """
        Returns True of `self` is a subset of `dataset`.

        Parameters
        ----------
        dataset : MaskDataset
            Dataset of which we want to know if `self` is a subset.

        Returns
        -------
        bool
            True of `self` is a subset of `dataset` else False.
        """
        # TODO maybe a better solution is suitable
        return dataset == self._dataset
