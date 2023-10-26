Gallery of examples
===================

This gallery contains several application examples for the ``nnbma`` package to illustrate diverse features.

For simplicity's sake, these notebooks will apply the approximation of a known analytical function rather than a complex physical model.

**Neural networks training:**

- ``training.ipynb``: neural network-based approximation of an analytical vectorial function
- ``masking.ipynb``: illustrate the use of a masked loss function to ignore unreliable labels
- ``operators.ipynb``: illustrate the use of pre and post processing operators

**Derivatives computing:**

- ``derivatives.ipynb``: illustrate how to compute the Jacobian or the Hessian of a neural network efficiently
- ``derivatives-time.ipynb``: illustrate the difference in computation time between the different ways of calculating the matrix

**Neural architectures:**

- ``fully-connected.ipynb``: classic multilayer perceptron
- ``densely-connected.ipynb``: perceptron architecture with dense shortcuts
- ``merging-network.ipynb`` : illustrate the use of the ``MergingNetwork`` network
- ``embedding-network.ipynb`` : illustrate the use of the ``EmbeddingNetwork`` network and the `AdditionalModule` layer

**Advanced neural networks features:**

- ``polynomial-expansion.ipynb``: illustrate the use of the ``PolynomialExpansion`` layer
- ``restrictable-layer.ipynb``: illustrate the use of the ``RestrictableLayer`` layer


.. toctree::
   :maxdepth: 1
   :caption: Gallery

   training
   masking
   operators

   derivatives

   fully-connected
   densely-connected
   merging-network
   embedding-network

   polynomial-expansion
   restrictable-layer
