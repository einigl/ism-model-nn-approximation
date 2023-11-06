from typing import List, Optional, Sequence, Union

# issue with import for python 3.9
try:
    from itertools import pairwise
except:
    from more_itertools import pairwise

import torch
from torch import Tensor, nn

from ..layers import AdditionalModule
from ..operators import Operator
from .neural_network import NeuralNetwork

__all__ = ["EmbeddingNetwork"]


class EmbeddingNetwork(NeuralNetwork):
    r"""Embedding neural network."""

    def __init__(
        self,
        subnetwork: NeuralNetwork,
        preprocessing: Union[None, AdditionalModule, List[AdditionalModule]] = None,
        postprocessing: Union[None, AdditionalModule, List[AdditionalModule]] = None,
        inputs_names: Optional[Sequence[str]] = None,
        outputs_names: Optional[Sequence[str]] = None,
        inputs_transformer: Optional[Operator] = None,
        outputs_transformer: Optional[Operator] = None,
        device: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        subnetwork : NeuralNetwork
            Base network.
        preprocessing : Union[None, AdditionalModule, List[AdditionalModule]], optional
            PyTorch operation to apply before ``subnetwork``, by default None.
        postprocessing : Union[None, AdditionalModule, List[AdditionalModule]], optional
            PyTorch operation to apply after ``subnetwork``, by default None.
        inputs_names : Optional[Sequence[str]], optional
            List of inputs names. None if the names have not been specified. By default None.
        outputs_names : Optional[Sequence[str]], optional
            List of outputs names. None if the names have not been specified. By default None.
        inputs_transformer : Optional[Operator], optional
            Transformation applied to the inputs before processing, by default None.
        outputs_transformer : Optional[Operator], optional
            Transformation applied to the outputs after processing, by default None.
        device : Optional[str], optional
            Device used ("cpu" or "cuda"), by default None (corresponds to "cpu").

        Raises
        ------
        TypeError
            All elements of preprocessing must be instances of AdditionalModule.
        TypeError
            All elements of postprocessing must be instances of AdditionalModule.
        """
        if preprocessing is None:
            preprocessing = []
        elif not isinstance(preprocessing, List):
            preprocessing = [preprocessing]
        if any([not isinstance(m, AdditionalModule) for m in preprocessing]):
            raise TypeError(
                "All elements of preprocessing must be instances of AdditionalModule"
            )

        n_inputs = subnetwork.input_features
        for m2, m1 in pairwise(([subnetwork] + preprocessing[::-1])):
            if (
                m2.input_features is not None
                and m1.output_features != m2.input_features
            ):
                raise ValueError(
                    f"{type(m1).__name__}.output_features ({m1.output_features}) doesn't match {type(m2).__name__}.input_features ({m1.input_features}"
                )
            if m1.input_features is not None:
                n_inputs = m1.input_features

        if postprocessing is None:
            postprocessing = []
        elif not isinstance(postprocessing, List):
            postprocessing = [postprocessing]
        if any([not isinstance(m, AdditionalModule) for m in postprocessing]):
            raise TypeError(
                "All elements of postprocessing must be instances of AdditionalModule"
            )

        n_outputs = subnetwork.output_features
        for m1, m2 in pairwise([subnetwork] + postprocessing):
            if (
                m2.output_features is not None
                and m1.output_features != m2.input_features
            ):
                raise ValueError(
                    f"{type(m1).__name__}.output_features ({m1.output_features}) doesn't match {type(m2).__name__}.input_features ({m2.input_features})"
                )
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
        self.preprocessing = nn.ModuleList(preprocessing)
        self.postprocessing = nn.ModuleList(postprocessing)

    def forward(self, x: Tensor) -> Tensor:
        y = x.clone()
        for op in self.preprocessing:
            y = op(y)
        y = self.subnetwork.forward(y)
        for op in self.postprocessing:
            y = op(y)
        return y
