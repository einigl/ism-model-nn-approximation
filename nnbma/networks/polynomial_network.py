from typing import Optional, Sequence, Union

from numpy import ndarray

from torch import Tensor

from ..layers import PolynomialExpansion
from ..preprocessing import Operator
from .neural_network import NeuralNetwork

__all__ = ["PolynomialNetwork"]


class PolynomialNetwork(NeuralNetwork):
    """
    Polynomial augmented features neural network.

    Attributes
    ----------
    att : type
        Description.
    """

    order: int
    n_poly_features: int

    def __init__(
        self,
        input_features: int,
        order: int,
        subnetwork: NeuralNetwork,
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
        super().__init__(
            input_features,
            subnetwork.output_features,
            inputs_names=inputs_names,
            outputs_names=outputs_names,
            inputs_transformer=inputs_transformer,
            outputs_transformer=outputs_transformer,
            device=device,
        )

        self.order = order
        self.n_poly_features = PolynomialExpansion.expanded_features(
            order, input_features
        )

        if self.n_poly_features != subnetwork.input_features:
            raise ValueError(
                f"The number of polynomial features ({self.n_poly_features}) does not match the input layer of the subnetwork ({subnetwork.input_features[0]})."
            )

        self.poly = PolynomialExpansion(
            input_features, order, self.device
        )
        self.subnetwork = subnetwork

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
        y_hat = self.poly(x)
        y_hat = self.subnetwork(y_hat)
        return y_hat
    
    def update_standardization(self, x: Union[Tensor, ndarray]) -> None:
        """
        Computes the mean and the standard deviation of the output of the polynomial expansion such that the expanded inputs are standardized.
        """
        self.poly.update_standardization(x)

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
        self.subnetwork.restrict_to_output_subset(output_subset)
