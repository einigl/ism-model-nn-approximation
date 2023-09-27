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
    input_features : int
        Dimension of input vector.
    output_features : int
        Dimension of output vector.
    inputs_names: List[str] | None
        List of inputs names. None if the names have not been specified.
    outputs_names: List[str] | None
        List of outputs names. None if the names have not been specified.
    inputs_transformer : Operator | None
        Transformation applied to the inputs before processing.
    outputs_transformer: Operator | None
        Transformation applied to the outputs after processing.
    device : str
        Device used ("cpu" or "cuda").
    """

    input_features: int
    output_features: int
    inputs_names: Optional[List[str]] = None
    outputs_names: Optional[List[str]] = None
    inputs_transformer: Optional[Operator] = None
    outputs_transformer: Optional[Operator] = None
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
        Set the device to use.

        Parameters
        ----------
        device : str
            Device to use ("cpu" or "cuda").
        """
        assert device in ["cuda", "cpu"], f"device = {device}"
        self.device = device

    def evaluate(self,
        x: np.ndarray,
        transform_inputs: bool=False,
        transform_outputs: bool=False,
    ) -> np.ndarray:
        """
        Process a batch of NumPy inputs.

        Parameters
        ----------
        x : ndarray
            Batch of inputs. Must be of shape N x self.n_inputs, where N is an arbitrary batch size.
        transform_inputs : bool, optional
            Specify if the data x has to be preprocessed (for instance standardized). If True, the preprocessing will be applied on the input data before being given to the network.
        transform_inputs : bool, optional
            Specify if the data x has to be postprocessed. If True, the postprocessing will be applied on the output data before being returned.

        Returns
        -------
        ndarray
            Batch of outputs of shape N x self.n_ouputs, where N is the inputs batch size.
        """
        if transform_inputs:
            if self.inputs_transformer is None:
                raise ValueError('transform_inputs cannot be True when self.inputs_transformer is None')
            x = self.inputs_transformer(x)

        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            y = self.forward(x)
        y = y.detach().cpu().numpy()

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
        Process a batch of NumPy ndarray or PyTorch Tensor inputs.

        Parameters
        ----------
        x : ndarray | Tensor
            Description.

        Returns
        -------
        type
            Description.
        """
        if isinstance(x, np.ndarray):
            return self.evaluate(x, False, False)
        else:
            with torch.no_grad():
                return self.forward(x)

    def train(self, mode: bool=True) -> "NeuralNetwork":
        """
        Set the current mode of the network (train or eval).

        Parameters
        ----------
        mode: bool, optional
            If True, activate the training mode. If False, activate the evaluation mode. Défault: True.

        Returns
        -------
        NeuralNetwork
            Instance of network.
        """
        super().train(mode)
        if mode:
            self.current_output_subset = self.outputs_names
            self.current_output_subset_indices = list(range(self.output_features))

    def restrict_to_output_subset(
        self,
        output_subset: Optional[Union[List[str], List[int]]]=None,
    ) -> None:
        """
        Restricts network outputs to those contained in `output_subset`.

        Parameters
        ----------
        output_subset : List[str] | List[int] | None, optional
            .Network outputs required. If None, no restriction is applied. Default: None.
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

    def _names_of_output_subset(
        self,
        output_subset: List[int]
    ) -> List[str]:
        """
        Returns the names of outputs corresponding to the indices list `output_subset`.

        Parameters
        ----------
        output_subset : List[int]
            Indices list.

        Returns
        -------
        List[str]
            Names of outputs.
        """
        if not isinstance(output_subset, (list, tuple)):
            raise TypeError("output_subset must be a list")
        if any(not isinstance(x, int) for x in output_subset):
            raise TypeError("output_subset must be a list of int")

        if not self._check_if_sublist(list(range(self.out_features)), output_subset):
            raise ValueError("input_subset is not a valid subset")

        return [self.outputs_names[k] for k in output_subset]

    def _indices_of_output_subset(
        self,
        output_subset: List[str]
    ) -> List[int]:
        """
        Returns the indices of outputs corresponding to the names list `output_subset`.

        Parameters
        ----------
        output_subset : List[str]
            Names list.

        Returns
        -------
        List[int]
            Indices of outputs.
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
    def _check_if_sublist(
        seq: List,
        subseq: List
    ) -> bool:
        """
        Returns True if all elements of `subseq` are also elements of `seq`. Else, return True. If `subseq` contains duplicates, the result will remain the same.

        Parameters
        ----------
        seq : List
            Reference list.
        subseq : List
            List to be checked for inclusion in `seq`.

        Returns
        -------
        bool
            Result.
        """
        return set(subseq) <= set(seq)

    @staticmethod
    def _indices_of_sublist(
        seq: List,
        subseq: List
    ) -> List[int]:
        """
        Returns the indices in `seq` of the elements of `subseq`. If `subseq` contains duplicates, the returned list will also contain duplicates.

        Parameters
        ----------
        seq : List
            Reference list.
        subseq : List
            List whose elements will be retrieved in `seq`.

        Returns
        -------
        List[int]
            Indices of elements of `subset` in `seq`.
        """
        if not NeuralNetwork._check_if_sublist(seq, subseq):
            raise ValueError("subseq is not a sublist of seq")
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
    ) -> Union[Tuple[int, Literal["B", "kB", "MB", "GB", "TB"]], str]:
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
            Unit ('B', 'kB', 'MB', 'GB', 'TB')
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
        """
        Saves the network for future use.

        Parameters
        ----------
        module_name: str
            Name of the directory in which the model will be saved.
        module_path: str | None
            Path to the previous directory.
        overwrite: bool
            If True, the save can overwrite a previous backup of the same name. If False, if such a backup exists, an error will be raised.
        """
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
        """
        Make a recursive save of a PyTorch module. This makes it possible to deal with cases where some of the parameters of a network are themselves networks, in which case this method avoids duplicates in the backup and saves memory space.

        Parameters
        ----------
        module: nn.Module
            Module to save.
        path:
            Path to save the Module.
        """
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
        """
        Returns True of obj is an object that need to be saved recursively, else False.
        
        Parameters
        ----------
        obj: object
            Any Python object.
        
        Returns
        -------
        bool
            True if `obj` needs to be save recursively, else False.
        """
        return isinstance(obj, NeuralNetwork)

    @staticmethod
    def _needs_json(obj: object) -> bool:
        """
        Returns True if the object `obj` must be saved in a JSON file.

        Parameters
        ----------
        obj: object
            Any Python object.

        Returns
        -------
        bool
            True if `obj` needs to be save in a JSON, else False.
        """
        if isinstance(obj, (bool, int, float, complex, str)):
            return True
        if isinstance(obj, (list, tuple)):
            return all(NeuralNetwork._needs_json(v) for v in obj)
        return False

    @staticmethod
    def _is_delegated(key: str, delegs: List[str]) -> bool:
        """
        Returns True the entry of key `key` can be delegated to a recursive save. The attributs that are to be save recursively are contained in `delegs`.

        Parameters
        ----------
        key: str
            Key of entry.
        delegs: List[str]
            List containing keys of attributes that will be save recursively.

        Returns
        -------
        bool
            True if `key` is the name of an attribute whose save can be delegated, else False.
        """
        for prefix in delegs:
            if key.startswith(prefix):
                return True
        return False

    @classmethod
    def load(
        self,
        module_name: str,
        module_path: Optional[str]=None
    ) -> "NeuralNetwork":
        """
        Load a network from a local save made using the `.save` method.

        Parameters
        ----------
        module_name: str
            Name of the directory in which the model has been saved.
        module_path: str | None
            Path to the previous directory.            
        """
        if module_path is None:
            path = module_name
        else:
            path = os.path.join(module_path, module_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} directory not exist")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory")

        net = NeuralNetwork._recursive_load(path)
        net.eval() # By default in evaluation mode

        return net

    @staticmethod
    def _recursive_load(path: str) -> "NeuralNetwork":
        """
        Make a recursive load of a PyTorch module.

        Parameters
        ----------
        path:
            Path to the module to load.
        """
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

    def copy(self) -> 'NeuralNetwork':
        """
        Returns a copy of self which perform exactly the same operation. The copy is detached from the original network so any modification of one doesn't modify the other.
        """
        d = {name: getattr(self, name) for name in list(signature(self.__init__).parameters)}

        for name in d:
            if NeuralNetwork._needs_recursion(d[name]):
                d[name] = d[name].copy()

        new = type(self)(**d)
        new.load_state_dict(self.state_dict())
        
        return new

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
