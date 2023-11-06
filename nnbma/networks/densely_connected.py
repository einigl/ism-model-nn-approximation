from math import ceil
from typing import Optional, Sequence, Union

from torch import Tensor, concat, nn

from ..layers import RestrictableLinear
from ..operators import Operator
from .neural_network import NeuralNetwork

__all__ = ["DenselyConnected"]


class DenselyConnected(NeuralNetwork):
    r"""Densely connected neural network. In such a network, the input of an hidden layer is the concatenation of the input and output of the previous layer. This `skip` operation permits to reduce the number of parameters to learn, to reuse intermediate computation results and to avoid gradient vanishing effects."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        n_layers: int,
        growing_factor: float,
        activation: nn.Module,
        batch_norm: bool = False,
        inputs_names: Optional[Sequence[str]] = None,
        outputs_names: Optional[Sequence[str]] = None,
        inputs_transformer: Optional[Operator] = None,
        outputs_transformer: Optional[Operator] = None,
        device: Optional[str] = None,
        last_restrictable: bool = True,
    ):
        """

        Parameters
        ----------
        input_features : int
            dimension of input vector.
        output_features : int
            dimension of output vector.
        n_layers : int
            number of layers in the network.
        growing_factor : float
            growing factor considered in the full network. The growing factor corresponds to the ratio of the output and input dimensions for one layer. For instance, ``growing_factor=1.0`` implies that the input of a hidden layer is twice that of the previous layer.
        activation : nn.Module
            activation function.
        batch_norm : bool, optional
            whether to use batch normalization during training, by default ``False``.
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
        last_restrictable : bool, optional
            whether the last layer is to be a RestrictableLinear layer, by default ``True``.
        """
        super().__init__(
            input_features,
            output_features,
            inputs_names=inputs_names,
            outputs_names=outputs_names,
            inputs_transformer=inputs_transformer,
            outputs_transformer=outputs_transformer,
            device=device,
        )

        self.n_layers = n_layers
        self.growing_factor = growing_factor

        self.activation = activation
        self.batch_norm = batch_norm

        self.last_restrictable = last_restrictable

        self.layers = nn.ModuleList()
        n_inputs = input_features
        self.layers_sizes = [input_features]
        for k in range(n_layers - 1):
            n_outputs = ceil(growing_factor * n_inputs)

            if batch_norm and k < n_layers - 2:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(n_inputs, n_outputs, device=self.device),
                        nn.BatchNorm1d(n_outputs, device=self.device),
                    )
                )
            else:
                self.layers.append(nn.Linear(n_inputs, n_outputs, device=self.device))

            n_inputs += n_outputs
            self.layers_sizes.append(n_inputs)

        if last_restrictable:
            self.output_layer = RestrictableLinear(
                n_inputs,
                output_features,
                outputs_names=self.outputs_names,
                device=self.device,
            )
        else:
            self.output_layer = nn.Linear(
                n_inputs,
                output_features,
                device=self.device,
            )
        self.layers_sizes.append(output_features)

    def forward(self, x: Tensor) -> Tensor:
        xk = x.clone()

        is1d = xk.ndim == 1
        if is1d:
            xk = xk.unsqueeze(0)

        for layer in self.layers:
            yk = layer(xk)
            yk = self.activation(yk)
            xk = concat((xk, yk), axis=-1)

        y_hat = self.output_layer(xk)

        if not self.last_restrictable:
            y_hat = y_hat[..., self.current_output_subset_indices]
        if is1d:
            y_hat = y_hat.squeeze(0)

        return y_hat

    def restrict_to_output_subset(
        self, output_subset: Optional[Union[Sequence[str], Sequence[int]]]
    ) -> None:
        super().restrict_to_output_subset(output_subset)
        if self.last_restrictable:
            self.output_layer.restrict_to_output_subset(
                self.current_output_subset_indices
            )
