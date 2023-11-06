from itertools import accumulate, chain
from typing import Optional, Sequence, Union
from warnings import warn

import torch
from torch import Tensor, nn

from ..operators import Operator
from .neural_network import NeuralNetwork

__all__ = ["MergingNetwork"]


class MergingNetwork(NeuralNetwork):
    r"""Utility class to run a set of neural networks in parallel to predict distinct sets of outputs."""

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

        Parameters
        ----------
        subnetworks : Sequence[NeuralNetwork]
            set of neural networks to be run in parallel to predict distinct sets of outputs.
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
            The ``subnetworks`` argument must be a sequence of NeuralNetwork instances.
        ValueError
            All the elements of ``subnetworks`` must have the same number of inputs.
        ValueError
            Incompatible ``inputs_names`` among ``subnetworks``.
        ValueError
            No element of ``subnetworks`` can be None when outputs_names is not None.
        ValueError
            Some elements of ``subnetworks`` have the same outputs.
        ValueError
            Some elements of ``outputs_names`` cannot be found in the outputs of any element of ``subnetworks``.
        """
        # Subnetworks
        if any([not isinstance(net, NeuralNetwork) for net in subnetworks]):
            raise TypeError("subnetworks must be a sequence of NeuralNetwork")

        # Inputs
        n_inputs = subnetworks[0].input_features
        if any([net.input_features != n_inputs for net in subnetworks]):
            raise ValueError(
                "All element of subnetworks must have the same number of inputs"
            )

        _inputs_names = [
            net.inputs_names for net in subnetworks if net.inputs_names is not None
        ]
        if any(name != _inputs_names[0] for name in _inputs_names):
            raise ValueError("Incompatible inputs_names among subnetworks")
        inputs_names = _inputs_names[0] if len(_inputs_names) > 0 else None

        # Outputs
        n_outputs = sum([net.output_features for net in subnetworks])
        if outputs_names is not None and any(
            [net.outputs_names is None for net in subnetworks]
        ):
            raise ValueError(
                "No element of subnetwork can be None when outputs_names is not None"
            )

        if outputs_names is None:
            if all([net.outputs_names is not None for net in subnetworks]):
                outputs_names = list(chain(*[net.outputs_names for net in subnetworks]))
            self.indices = list(
                range(sum([net.output_features for net in subnetworks]))
            )
        else:
            _outputs_names = list(chain(*[net.outputs_names for net in subnetworks]))
            if len(set(_outputs_names)) != len(_outputs_names):
                raise ValueError("Some subnetworks have the same outputs")
            if not (set(_outputs_names) >= set(outputs_names)):
                raise ValueError(
                    "Some elements of outputs_names cannot be found in any subnetworks"
                )
            if set(_outputs_names) != set(outputs_names):
                warn("Some subnetworks outputs are not retrieved")
            if len(_outputs_names) != len(outputs_names):
                warn("There are duplicates in outputs_names")
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
        res = []

        for net in self.subnetworks:
            res.append(net.forward(x))

        return torch.concat(res, dim=-1)[..., self.indices]

    def restrict_to_output_subset(
        self, output_subset: Optional[Union[Sequence[str], Sequence[int]]]
    ) -> None:
        super().restrict_to_output_subset(output_subset)

        if isinstance(output_subset[0], int):
            net_start_indices = list(
                accumulate([0] + [net.output_features for net in self.subnetworks][:-1])
            )
            net_end_indices = list(
                accumulate([net.output_features for net in self.subnetworks])
            )

        for net in self.subnetworks:
            if output_subset is None:
                net_output_subset = None
            elif isinstance(output_subset[0], str):
                net_output_subset = [
                    name for name in output_subset if name in net.outputs_names
                ]
            else:
                start, end = next(net_start_indices), next(net_end_indices)
                real_output_subset = [
                    self.indices[idx]
                    for idx in output_subset
                    if idx >= start and idx < end
                ]
                net_output_subset = [idx - start for idx in real_output_subset]
            net.restrict_to_output_subset(net_output_subset)
