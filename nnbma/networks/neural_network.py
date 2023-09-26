import json
import os
import pickle
import shutil
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from inspect import signature
from typing import List, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import torch
from torch import nn

from ..preprocessing import Operator

__all__ = ["NeuralNetwork"]


class NeuralNetwork(nn.Module, ABC):
    """
    Neural network abstract class.

    Attributes
    ----------
    att : type
        Description.
    """

    input_features: int
    output_features: int
    device: str
    current_output_subset: List[str]
    current_output_subset_indices: List[int]

    def __init__(
        self,
        input_features: int,
        output_features: int,
        inputs_names: Optional[List[str]] = None,
        outputs_names: Optional[List[str]] = None,
        inputs_transformer: Optional[Operator] = None,
        outputs_transformer: Optional[Operator] = None,
        device: Optional[str] = None,
    ):
        """
        Initializer.

        Parameters
        ----------
        param : type
            Description.
        """
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        if inputs_names is not None and len(inputs_names) != input_features:
            raise ValueError(f"inputs_names of length {len(inputs_names)} is incompatible with {input_features} input_features")
        if outputs_names is not None and len(outputs_names) != output_features:
            raise ValueError(f"outputs_names of length {len(outputs_names)} is incompatible with {output_features} outputs_features")

        if inputs_names is not None and len(set(inputs_names)) != len(inputs_names):
            raise ValueError("inputs_names has duplicates")
        if outputs_names is not None and len(set(outputs_names)) != len(outputs_names):
            raise ValueError("outputs_names has duplicates")

        self.inputs_names = inputs_names
        self.outputs_names = outputs_names

        self.inputs_transformer = inputs_transformer
        self.outputs_transformer = outputs_transformer

        self.current_output_subset = outputs_names
        self.current_output_subset_indices = list(range(output_features))

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.eval()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def set_device(self, device: str) -> None:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        assert device in ["cuda", "cpu"], f"device = {device}"
        self.device = device

    def evaluate(self,
        x: np.ndarray,
        transform_inputs: bool=False,
        transform_outputs: bool=False,
    ) -> np.ndarray:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """

        if transform_inputs:
            if self.inputs_transformer is None:
                raise ValueError('transform_inputs cannot be True when self.inputs_transformer is None')
            x = self.inputs_transformer(x)

        x = torch.from_numpy(x).double().to(self.device)
        with torch.no_grad():
            y = self.forward(x)
        y = y.detach().cpu().numpy().astype(np.float64)

        if transform_outputs:
            if self.outputs_transformer is None:
                raise ValueError('transform_outputs cannot be True when self.outputs_transformer is None')
            y = self.outputs_transformer(y)

        return y

    @overload
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @overload
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        if isinstance(x, np.ndarray):
            return self.evaluate(x)
        else:
            return self.forward(x)

    def train(self, mode: bool = True) -> "NeuralNetwork":
        super().train(mode)
        if mode:
            self.current_output_subset = self.outputs_names
            self.current_output_subset_indices = list(range(self.output_features))

    def restrict_to_output_subset(
        self, output_subset: Optional[Union[List[str], List[int]]]
    ) -> None:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        if self.training:
            raise PermissionError(
                "You're not able to restrict the outputs when Module mode is train"
            )

        if output_subset is None:
            self.current_output_subset = self.inputs_names
            self.current_output_subset_indices = list(range(self.output_features))
            return

        if not isinstance(output_subset, List):
            raise TypeError("output_subset must be a list or None")
        if len(output_subset) == 0:
            raise ValueError("output_subset must not be empty")

        if all(isinstance(x, int) for x in output_subset):
            self.current_output_subset = self._names_of_output_subset(output_subset)
            self.current_output_subset_indices = output_subset
        elif all(isinstance(x, str) for x in output_subset):
            self.current_output_subset = output_subset
            self.current_output_subset_indices = self._indices_of_output_subset(
                output_subset
            )
        else:
            raise TypeError("output_subset must be a list of int or a list of str")

    def _names_of_output_subset(self, output_subset: List[int]) -> List[int]:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        if not isinstance(output_subset, (list, tuple)):
            raise TypeError("output_subset must be a list")
        if any(not isinstance(x, int) for x in output_subset):
            raise TypeError("output_subset must be a list of int")

        if not self._check_if_sublist(list(range(self.out_features)), output_subset):
            raise ValueError("input_subset is not a valid subset")

        return [self.outputs_names[k] for k in output_subset]

    def _indices_of_output_subset(self, output_subset: List[str]) -> List[int]:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        if not isinstance(output_subset, (list, tuple)):
            raise TypeError("output_subset must be a list")
        if any(not isinstance(x, str) for x in output_subset):
            raise TypeError("output_subset must be a list of int")

        if self.outputs_names is None:
            raise TypeError(
                "output_subset cannot be a list of str when self.outputs_names is None"
            )
        if not self._check_if_sublist(self.outputs_names, output_subset):
            raise ValueError("output_subset is not a valid subset")

        return self._indices_of_sublist(self.outputs_names, output_subset)

    @staticmethod
    def _check_if_sublist(seq: List, subseq: List) -> bool:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        return set(subseq) <= set(seq)

    @staticmethod
    def _indices_of_sublist(seq: List, subseq: List) -> List[int]:
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        index_dict = dict((value, idx) for idx, value in enumerate(seq))
        return [index_dict[value] for value in subseq]  # Remark: the result is ordered

    def count_parameters(self, learnable_only: bool = True) -> int:
        """
        Returns the number of parameters of the module.
        If `learnable_only` is True, then this function returns the number of parameters whose has a `requires_grad = True` property.
        If `learnable_only` is False, then this function returns the number of parameters, independently to their `requires_grad` property.

        Parameters
        ----------
        learnable_only : bool, optional
            Indicates the the type of parameter to count. Defaults to True.

        Returns
        -------
        int
            Number of parameters.
        """
        self.train()
        if learnable_only:
            count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        count = sum(p.numel() for p in self.parameters())
        self.eval()
        return count

    def count_bytes(
        self,
        learnable_only: bool = True,
        display: bool = False,
    ) -> Union[Tuple[int, Literal["b", "kb", "Mb", "Gb", "Tb", "Pb"]], str]:
        """
        Returns the number of parameters of the module.
        If `learnable_only` is True, then this function returns the number of parameters whose has a `requires_grad = True` property.
        If `learnable_only` is False, then this function returns the number of parameters, independently to their `requires_grad` property.

        Parameters
        ----------
            learnable_only (bool, optional): Indicates the the type of parameter to count. Defaults to True.

        Returns
        -------
        int
            Number of parameters.
        str
            Unit ('b', 'kb', 'Mb', 'Gb', 'Tb')
        """
        self.train()
        size = 0.0
        for p in self.parameters():
            if p.requires_grad or not learnable_only:
                size += (
                    p.numel() * p.element_size()
                )  # Does not take into consideration the Python object size which can be obtain using sys.getsizeof()
        self.eval()

        for (v, u) in [(1e0, "B"), (1e3, "kB"), (1e6, "MB"), (1e9, "GB"), (1e12, "TB")]:
            if size < 1e3 * v:
                if display:
                    return f"{size / v:.2f} {u}"
                else:
                    return (size / v, u)

    def time(self, n: int, repeat: int) -> Tuple[float, float, float]:
        """
        Compute the evaluation time of the model for a batch of `n` inputs. Returns the average, min and max durations (in sec) over `repeat` iterations.
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, not {type(n)}")
        if not isinstance(repeat, int):
            raise TypeError(f"repeat must be an integer, not {type(repeat)}")
        times = []
        for it in range(repeat):
            x = torch.normal(0., 1., size=(n, self.input_features))
            tic = time.time()
            self.forward(x)
            toc = time.time()
            times.append(toc-tic)
        return sum(times)/repeat, min(times), max(times)

    def save(self,
        module_name: str,
        module_path: Optional[str] = None,
        overwrite: bool = True
    ) -> None:
        if module_path is not None and not os.path.isdir(module_path):
            os.mkdir(module_path)

        if module_path is None:
            path = module_name
        else:
            path = os.path.join(module_path, module_name)

        if not os.path.exists(path):
            pass
        elif not overwrite:
            raise FileExistsError(f'{path} already exists')
        else:
            for _, __, f in os.walk(path):
                if not all(os.path.isdir(file) or file.endswith(('.json', '.pkl', '.pth')) for file in f):
                    raise ValueError(f"{path} directory cannot be overwritten because it doesn't seem to be a NeuralNetwork save directory")
            shutil.rmtree(path)

        NeuralNetwork._recursive_save(self, path)

    @staticmethod
    def _recursive_save(
        module: nn.Module,
        path: str,
    ) -> None:
        os.mkdir(path)
        template = os.path.join(path, '{}')

        args = list(signature(module.__init__).parameters)
        with open(template.format("init.pkl"), "wb") as f:
            pickle.dump((type(module), args), f)

        delegs = []
        for arg in args:
            obj = getattr(module, arg)
            if NeuralNetwork._needs_recursion(obj):
                NeuralNetwork._recursive_save(obj, os.path.join(path, arg))
                delegs.append(arg)
            else:
                if NeuralNetwork._needs_json(obj):
                    with open(template.format(f"{arg}.json"), "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=4)
                else :
                    with open(template.format(f"{arg}.pkl"), "wb") as f:
                        pickle.dump(obj, f)

        sd = module.state_dict()
        sd = OrderedDict([(key, val) for key, val in sd.items() if not NeuralNetwork._is_delegated(key, delegs)])

        torch.save(sd, template.format('state_dict.pth'))

    @staticmethod
    def _needs_recursion(obj: object) -> bool:
        """ Returns True of obj is an object that need to be saved recursively, else False """
        return isinstance(obj, NeuralNetwork)

    @staticmethod
    def _needs_json(obj: object) -> bool:
        """ Returns True if the object `obj` must be saved in a JSON file. """
        if isinstance(obj, (bool, int, float, complex, str)):
            return True
        if isinstance(obj, (list, tuple)):
            return all(NeuralNetwork._needs_json(v) for v in obj)
        return False

    @staticmethod
    def _is_delegated(key: str, delegs: List[str]) -> bool:
        for prefix in delegs:
            if key.startswith(prefix):
                return True
        return False

    @classmethod
    def load(self, module_name: str, module_path: Optional[str] = None) -> "NeuralNetwork":
        if module_path is None:
            path = module_name
        else:
            path = os.path.join(module_path, module_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} directory not exist")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory")

        return NeuralNetwork._recursive_load(path)

    @staticmethod
    def _recursive_load(path: str) -> "NeuralNetwork":
        template = os.path.join(path, '{}')

        with open(template.format("init.pkl"), "rb") as f:
            type_module, args = pickle.load(f)

        d = {}
        for arg in args:
            if os.path.exists(template.format(f"{arg}.json")):
                with open(template.format(f"{arg}.json"), "r") as f:
                    d.update({arg: json.load(f)})
            elif os.path.exists(template.format(f"{arg}.pkl")):
                with open(template.format(f"{arg}.pkl"), "rb") as f:
                    d.update({arg: pickle.load(f)})
            elif os.path.isdir(template.format(arg)):
                d.update({arg: NeuralNetwork._recursive_load(template.format(arg))})
            else:
                raise RuntimeError("Should never been here.")

        module: nn.Module = type_module(**d)

        sd = module.state_dict()
        sd.update(torch.load(template.format('state_dict.pth')))
        module.load_state_dict(sd)

        return module

    def copy(self):
        """ TODO """
        d = {name: getattr(self, name) for name in list(signature(self.__init__).parameters)}
        return type(self)(**d) # TODO: state dict

    def __str__(self) -> str:
        d = list(signature(self.__init__).parameters)
        descr = f'{type(self).__name__}:\n'
        for arg in d:
            obj = getattr(self, arg)
            if isinstance(obj, list) and len(obj) > 6:
                obj = obj[:6] + ['...']
            elif isinstance(obj, tuple):
                obj = obj[:6] + ('...', )
            descr += f'\t{arg}: {obj}\n'
        return descr
