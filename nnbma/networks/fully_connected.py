from typing import Iterable, Optional, Sequence, Union

from torch import Tensor, nn

from ..layers import RestrictableLinear
from ..preprocessing import Operator

from .neural_network import NeuralNetwork

__all__ = ["FullyConnected"]


class FullyConnected(NeuralNetwork):
    """
    Fully connected neural network.

    Attributes
    ----------
    att : type
        Description.
    """

    def __init__(
        self,
        layers_sizes: Iterable[int],
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
            layers_sizes[0],
            layers_sizes[-1],
            inputs_names=inputs_names,
            outputs_names=outputs_names,
            inputs_transformer=inputs_transformer,
            outputs_transformer=outputs_transformer,
            device=device,
        )

        self.layers_sizes = layers_sizes

        self.activation = activation
        self.batch_norm = batch_norm

        self.last_restrictable = last_restrictable

        self.layers = nn.ModuleList()
        for k in range(len(layers_sizes) - 2):
            if batch_norm:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(
                            self.layers_sizes[k],
                            self.layers_sizes[k + 1],
                            bias=False,
                            device=self.device,
                        ),
                        nn.BatchNorm1d(
                            self.layers_sizes[k + 1],
                            device=self.device,
                        ),
                    )
                )
            else:
                self.layers.append(
                    nn.Linear(
                        self.layers_sizes[k],
                        self.layers_sizes[k + 1],
                        bias=True,
                        device=self.device,
                    )
                )

        if last_restrictable:
            self.output_layer = RestrictableLinear(
                layers_sizes[-2],
                layers_sizes[-1],
                outputs_names=outputs_names,
                device=self.device,
            )
        else:
            self.output_layer = nn.Linear(
                layers_sizes[-2],
                layers_sizes[-1],
                device=self.device,
            )

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
        y_hat = x.clone()
        
        is1d = y_hat.ndim == 1
        if is1d:
            y_hat = y_hat.unsqueeze(0)

        for layer in self.layers:
            y_hat = layer(y_hat)
            y_hat = self.activation(y_hat)

        y_hat = self.output_layer(y_hat)

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
