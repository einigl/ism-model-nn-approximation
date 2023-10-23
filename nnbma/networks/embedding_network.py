from typing import Optional, Sequence, Union, List

# issue with import for python 3.9
try:
    from itertools import pairwise
except:
    from more_itertools import pairwise

import torch
from torch import Tensor, nn

from ..layers import AdditionalModule
from ..preprocessing import Operator
from .neural_network import NeuralNetwork

__all__ = ["EmbeddingNetwork"]


class EmbeddingNetwork(NeuralNetwork):
    """
    Embedding neural network.

    Attributes
    ----------
    subnetwork : NeuralNetork
        Base network.
    preprocessing: Optional[nn.Module]
        PyTorch operation to apply before `subnetwork`.
    postprocessing: Optional[nn.Module]
        PyTorch operation to apply after `subnetwork`.
    """

    subnetwork: NeuralNetwork
    preprocessing: nn.Sequential
    postprocessing: nn.Sequential

    def __init__(
        self,
        subnetwork: NeuralNetwork,
        preprocessing: Union[None, AdditionalModule, List[AdditionalModule]]=None,
        postprocessing: Union[None, AdditionalModule, List[AdditionalModule]]=None,
        inputs_names: Optional[Sequence[str]] = None,
        outputs_names: Optional[Sequence[str]] = None,
        inputs_transformer: Optional[Operator] = None,
        outputs_transformer: Optional[Operator] = None,
        device: Optional[str] = None,
    ):
        """
        Initializer.

        Parameters
        ----------
        subnetwork: NeuralNetork
            Base network.
        preprocessing: None | AdditionalModule | List[AdditionalModule]
            PyTorch operation to apply before `subnetwork`.
        postprocessing: None | AdditionalModule | List[AdditionalModule]
            PyTorch operation to apply after `subnetwork`.
        """
        if preprocessing is None:
            preprocessing = []
        elif not isinstance(preprocessing, List):
            preprocessing = [preprocessing]
        if any([not isinstance(m, AdditionalModule) for m in preprocessing]):
            raise TypeError("All elements of preprocessing must be instances of AdditionalModule")

        n_inputs = subnetwork.input_features
        for m2, m1 in pairwise(([subnetwork] + preprocessing[::-1])):
            if m2.input_features is not None and m1.output_features != m2.input_features:
                raise ValueError(f"{type(m1).__name__}.output_features ({m1.output_features}) doesn't match {type(m2).__name__}.input_features ({m1.input_features}")
            if m1.input_features is not None:
                n_inputs = m1.input_features

        if postprocessing is None:
            postprocessing = []
        elif not isinstance(postprocessing, List):
            postprocessing = [postprocessing]
        if any([not isinstance(m, AdditionalModule) for m in postprocessing]):
            raise TypeError("All elements of postprocessing must be instances of AdditionalModule")

        n_outputs = subnetwork.output_features
        for m1, m2 in pairwise([subnetwork] + postprocessing):
            if m2.output_features is not None and m1.output_features != m2.input_features:
                raise ValueError(f"{type(m1).__name__}.output_features ({m1.output_features}) doesn't match {type(m2).__name__}.input_features ({m2.input_features})")
            if m2.output_features is not None:
                n_outputs = m2.output_features

        super().__init__(
            n_inputs,
            n_outputs,
            inputs_names=inputs_names,
            outputs_names=outputs_names,
            inputs_transformer=inputs_transformer,
            outputs_transformer=outputs_transformer,
            device=device,
        )

        self.subnetwork = subnetwork
        self.preprocessing = nn.Sequential(*preprocessing)
        self.postprocessing = nn.Sequential(*postprocessing)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the output of the network for a batch of inputs `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        y = self.preprocessing(x)
        y = self.subnetwork(y)
        y = self.postprocessing(y)
        return y
