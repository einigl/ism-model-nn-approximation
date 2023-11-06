from typing import Optional, Sequence, Union

from numpy import ndarray
from torch import Tensor

from ..layers import PolynomialExpansion
from ..operators import Operator
from .neural_network import NeuralNetwork

__all__ = ["PolynomialNetwork"]


class PolynomialNetwork(NeuralNetwork):
    r"""Neural network with a polynomial expansion as a first layer."""

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
        r"""

        Parameters
        ----------
        input_features : int
            Dimension of input vector.
        order : int
            order of the polynomial expansion.
        subnetwork : NeuralNetwork
            network to be placed after the polynomial expansion.
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
        ValueError
            The number of polynomial features does not match the input layer of the subnetwork.
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

        self.poly = PolynomialExpansion(input_features, order, self.device)
        self.subnetwork = subnetwork

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.poly.forward(x)
        y_hat = self.subnetwork.forward(y_hat)
        return y_hat

    def update_standardization(self, x: Union[Tensor, ndarray]) -> None:
        """Applies the ``update_standardization`` method of the PolynomialExpansion first layer, i.e., updates the standardization parameters for the outputs of the polynomial expansion.

        Parameters
        ----------
        x : Union[Tensor, ndarray]
            input tensor.
        """
        self.poly.update_standardization(x)

    def restrict_to_output_subset(
        self, output_subset: Optional[Union[Sequence[str], Sequence[int]]]
    ) -> None:
        super().restrict_to_output_subset(output_subset)
        self.subnetwork.restrict_to_output_subset(output_subset)
