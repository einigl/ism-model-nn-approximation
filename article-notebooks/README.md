# Article notebooks

This directory includes several Python notebooks to illustrate or reproduce the results presented in the article.

## Neural networks-based emulation of the Meudon PDR code

- `nn-regression-fc.ipynb`: emulate the code with a fully connected NN
- `nn-regression-dense.ipynb`: emulate the code with a densely connected NN
- `nn-regression-fc-clustering.ipynb`: emulate the code with several fully connected NN dedicated to subsets of lines
- `nn-regression-dense-clustering.ipynb`: emulate the code with several densely connected NN dedicated to subsets of lines

## Other notebooks

- `plotting.ipynb`: plot profile and slices of the input parameters space to visualize the emulation
- `clustering.ipynb`: cluster the lines in different subsets based on similarity
- `derivatives.ipynb`: compute the n-th order derivatives of lines intensities with respect to the inputs

__Note:__ Some features, such as multiple simultaneous progress bars, don't work well in an iPython environment. If you feel the need, it may be worth exporting certain `.ipynb` notebooks as `.py` Python files.