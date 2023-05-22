from math import ceil
from typing import Optional, Sequence, Union

from torch import Tensor, concat, nn

from ..preprocessing import Operator
from ..layers import RestrictableLinear

from .neural_network import NeuralNetwork

__all__ = ["DenselyConnected"]


class DenselyConnected(NeuralNetwork):
    """
    Densely connected neural network.

    Attributes
    ----------
    att : type
        Description.
    """

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
        Initializer.

        Parameters
        ----------
        param : type
            Description.
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
            self.layers_sizes.append(n_outputs)

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

        # self.layers.to(self.device)

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

        # print('layers_sizes')
        # print(self.layers_sizes) # TODO
        # print(sum(self.layers_sizes[:-1]))

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
        if self.last_restrictable:
            self.output_layer.restrict_to_output_subset(
                self.current_output_subset_indices
            )
