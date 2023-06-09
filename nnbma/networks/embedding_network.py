from typing import Optional, Sequence, Union
from warnings import warn
from itertools import accumulate, chain

import torch
from torch import Tensor, nn

from ..preprocessing import Operator

from .neural_network import NeuralNetwork

__all__ = ["EmbeddingNetwork"]


class EmbeddingNetwork(NeuralNetwork):
    """
    Embedding neural network.

    Attributes
    ----------
    att : type
        Description.
    """

    def __init__(
        self,
        subnetworks: Sequence[NeuralNetwork],
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
        param : type
            Description.
        """
        n_inputs = subnetworks[0].input_features
        n_outputs = sum([net.output_features for net in subnetworks])

        if any([not isinstance(net, NeuralNetwork) for net in subnetworks]):
            raise ValueError("subnetworks must be a sequence of NeuralNetwork")

        if outputs_names is not None and any([net.outputs_names is None for net in subnetworks]):
            raise ValueError("No element of subnetwork can be None when outputs_names is not None")

        if outputs_names is None:
            if all([net.outputs_names is not None for net in subnetworks]):
                self.outputs_names = list(chain(*[net.outputs_names for net in subnetworks]))
            self.indices = list(range(sum([net.output_features for net in subnetworks])))            
        else:
            _outputs_names = list(chain(*[net.outputs_names for net in subnetworks]))
            if len(set(_outputs_names)) != len(_outputs_names):
                raise ValueError("Some subnetworks have the same outputs")
            if not (set(_outputs_names) >= set(outputs_names)):
                raise ValueError("Some elements of outputs_names are not findable in any subnetworks")
            if set(_outputs_names) != set(outputs_names):
                warn("Some subnetworks outputs are not retrieved")
            self.indices = [_outputs_names.index(name) for name in outputs_names]

        super().__init__(
            n_inputs,
            n_outputs,
            inputs_names=inputs_names,
            outputs_names=outputs_names,
            inputs_transformer=inputs_transformer,
            outputs_transformer=outputs_transformer,
            device=device,
        )

        self.subnetworks = nn.ModuleList(subnetworks)

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
        res = []

        for net in self.subnetworks:
            res.append(net.forward(x))

        return torch.concat(res, dim=-1)[..., self.indices]

    def restrict_to_output_subset(
        self, output_subset: Optional[Union[Sequence[str], Sequence[int]]]
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
        super().restrict_to_output_subset(output_subset)

        if isinstance(output_subset[0], int):
            net_start_indices = list(accumulate([0] + [net.output_features for net in self.subnetworks][:-1]))
            net_end_indices = list(accumulate([net.output_features for net in self.subnetworks]))

        for net in self.subnetworks:
            if output_subset is None:
                net_output_subset = None
            elif isinstance(output_subset[0], str):
                net_output_subset = [name for name in output_subset if name in net.outputs_names]
            else:
                start, end = next(net_start_indices), next(net_end_indices)
                real_output_subset = [self.indices[idx] for idx in output_subset if idx >= start and idx < end]
                net_output_subset = [idx-start for idx in real_output_subset]
            net.restrict_to_output_subset(net_output_subset)
