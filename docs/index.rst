.. nnbma documentation master file, created by
   sphinx-quickstart on Mon Oct 23 19:44:16 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nnbma's documentation
=================================

Neural network-based model approximation ``nnbma`` is a Python package that handle the creation and the training of neural networks to approximate numerical models.
In [1], it was designed and used to derive an approximation of the Meudon PDR code, a complex astrophysical numerical code.

============
Installation
============

To build your own neural network for your numerical model, we recommend installing the package.
The package can be installed with ``pip``:

``pip install nnbma``

To reproduce the results from [1], clone the repo with

``git clone git@github.com:einigl/ism-model-nn-approximation.git``

Alternatively, you can also download a zip file.

This package relies on ``pytorch`` to build neural networks.
It enables to evaluate any neural network, its gradient, and its Hessian matrix efficiently.

If you do not have a Python environment compatible with the above dependencies, we advise you to create a specific conda environment to use this code.
See https://conda.io/projects/conda/en/latest/user-guide/ for more information on conda environment management.

.. toctree::
   :maxdepth: 4
   :caption: Contents

   modules

.. toctree::
   :maxdepth: 2

   gallery-examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Reference
=========

[1] Palud, P. & Einig, L. & Le Petit, F. & Bron, E. & Chainais, P. & Chanussot, J. & Pety, J. & Thouvenin, P.-A. & Languignon, D. & Beslić, I. & G. Santa-Maria, M. & Orkisz, J.H. & Ségal, L. & Zakardjian, A. & Bardeau, S. & Gerin, M. & Goicoechea, J.R. & Gratier, P. & Guzman, V. (2023). Neural network-based emulation of interstellar medium models. Astronomy & Astrophysics. 10.1051/0004-6361/202347074.
