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
            Set of neural networks to be run in parallel to predict distinct sets of outputs.
        inputs_names : Optional[Sequence[str]], optional
            List of inputs names. None if the names have not been specified. By default None.
        outputs_names : Optional[Sequence[str]], optional
            List of outputs names. None if the names have not been specified.
            Must be coherent with subnetworks names. If not None, all subnetworks must have a non-None ``outputs_names`` attribute.
            By default None.
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

        if inputs_names is None:
            inputs_names = _inputs_names[0] if len(_inputs_names) > 0 else None
        elif subnetworks[0].input_features != len(inputs_names):
            raise ValueError(
                f"Length of inputs_names ({len(inputs_names)}) is incompatible with subnetworks number of inputs ({subnetworks[0].input_features})"
            )

        # Outputs
        n_outputs = sum([net.output_features for net in subnetworks])

        # Check whether some subnetworks have common outputs names
        _outputs_names = list(
            chain(
                *[
                    net.outputs_names
                    for net in subnetworks
                    if net.outputs_names is not None
                ]
            )
        )
        if len(set(_outputs_names)) != len(_outputs_names):
            raise ValueError("Some subnetworks have the same outputs")
        del _outputs_names

        # Check whether all subnetworks have outputs names
        if all([net.outputs_names is not None for net in subnetworks]):
            _outputs_names = list(chain(*[net.outputs_names for net in subnetworks]))
        else:
            _outputs_names = None

        # Other checks if outputs_names is not None:
        if outputs_names is not None:
            # Errors
            if any([net.outputs_names is None for net in subnetworks]):
                raise ValueError(
                    "No subnetwork output_names attribute can be None when outputs_names is not None"
                )
            if not set(_outputs_names) >= set(outputs_names):
                raise ValueError(
                    "Some elements of outputs_names cannot be found in any subnetworks"
                )
            # Warnings
            if set(_outputs_names) > set(outputs_names):
                warn("Some subnetworks outputs are not propagated")
            if len(_outputs_names) > len(outputs_names):
                warn("There are duplicates in outputs_names")

        # Extrapolate outputs names if possible
        if outputs_names is None and _outputs_names is not None:
            outputs_names = _outputs_names

        # Derive indices
        if outputs_names is not None:
            self.indices = [_outputs_names.index(name) for name in outputs_names]
        else:
            self.indices = list(
                range(sum([net.output_features for net in subnetworks]))
            )
        self.current_indices = self.indices.copy()

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

        # Ensure that no subnetworks is restricted
        for net in self.subnetworks:
            if net.restricted:
                raise ValueError("All subnetworks must have unrestricted outputs")

    def forward(self, x: Tensor) -> Tensor:
        res = []
        for net in self.subnetworks:
            res.append(net.forward(x))

        return torch.concat(res, dim=-1)[..., self.current_indices]

    def restrict_to_output_subset(
        self, output_subset: Optional[Union[Sequence[str], Sequence[int]]] = None
    ) -> None:
        super().restrict_to_output_subset(output_subset)

        if output_subset is None:
            self.current_indices = self.indices.copy()
            for net in self.subnetworks:
                net.restrict_to_output_subset(None)
            return
        if len(output_subset) == 0:
            self.current_indices = []
            for net in self.subnetworks:
                net.restrict_to_output_subset([])
            return

        # Case of a list of strings
        if isinstance(output_subset[0], str):
            _subnets_subsets = []
            for net in self.subnetworks:
                _subset = [name for name in output_subset if name in net.outputs_names]
                _subset = list(
                    set(_subset)
                )  # May change the order but it doesn't matter
                net.restrict_to_output_subset(_subset)
                _subnets_subsets += _subset
            self.current_indices = [
                _subnets_subsets.index(name) for name in output_subset
            ]
            return

        # Case of a list of integers
        _starts = list(
            accumulate(
                [net.output_features for net in self.subnetworks[:-1]], initial=0
            )
        )
        _ends = [
            i - 1 for i in accumulate([net.output_features for net in self.subnetworks])
        ]

        if isinstance(output_subset[0], int):
            _effective_output_subset = [self.indices[i] for i in output_subset]
            _subnets_subsets = []
            for net, start, end in zip(self.subnetworks, _starts, _ends):
                _subset = [i for i in _effective_output_subset if start <= i <= end]
                _subset_wo_dup = list(set(_subset))
                net.restrict_to_output_subset([i - start for i in _subset_wo_dup])
                _subnets_subsets += [
                    _subset_wo_dup.index(i) + len(_subnets_subsets) for i in _subset
                ]
            self.current_indices = _subnets_subsets
