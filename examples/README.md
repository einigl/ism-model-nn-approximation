# Example notebooks

This directory contains several Python notebooks providing documentation for the `nnbma` package and illustrating the different existing features.

For simplicity's sake, these notebooks will apply the approximation of a known analytical function rather than a complex physical model. To reproduce the results of the paper on the Meudon PDR code, please refer to the notebooks in the `article-notebooks` directory.

Neural networks training:

- `training.ipynb`: neural network-based approximation of an analytical vectorial function

Derivatives computing:

- `derivatives.ipynb`: illustrate how to compute the Jacobian or the Hessian of a neural network efficiently
- `derivatives-time.ipynb`: illustrate the difference in computation time between the different ways of calculating the matrix

Neural networks features

- `polynomial-expansion.ipynb`: illustrate the use of the `PolynomialExpansion` layer
- `restrictable-layer.ipynb`: illustrate the use of the `RestrictableLayer` layer
- `merging-network.ipynb` : illustrate the use of the `MergingNetwork` network
- `embedding-network.ipynb` : illustrate the use of the `EmbeddingNetwork` network and the `AdditionalModule` layer

__Note:__ Some features, such as multiple simultaneous progress bars, don't work well in an iPython environment. If you feel the need, it may be worth exporting certain `.ipynb` notebooks as `.py` Python files.
