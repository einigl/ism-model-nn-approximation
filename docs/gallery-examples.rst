Gallery of examples
===================

This gallery contains several application examples for the ``nnbma`` package to illustrate diverse features.

For simplicity's sake, these notebooks will apply the approximation of a known analytical function rather than a complex physical model.

Neural networks training:

- `training`: neural network-based approximation of an analytical vectorial function

Derivatives computing:

- `derivatives`: illustrate how to compute the Jacobian or the Hessian of a neural network efficiently
- `derivatives-time`: illustrate the difference in computation time between the different ways of calculating the matrix

Neural networks features:

- `polynomial-expansion`: illustrate the use of the `PolynomialExpansion` layer
- `restrictable-layer`: illustrate the use of the `RestrictableLayer` layer
- `merging-network` : illustrate the use of the `MergingNetwork` network
- `embedding-network` : illustrate the use of the `EmbeddingNetwork` network and the `AdditionalModule` layer


.. toctree::
   :maxdepth: 1
   :caption: Gallery

   training
   derivatives
   polynomial-expansion
