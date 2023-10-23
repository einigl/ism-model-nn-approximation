# Meudon PDR code dataset

To reproduce the neural networks described in [1], this directory should contain data downloaded from <https://ism.obspm.fr/files/ArticleData/2023_Palud_Einig/2023_Palud_Einig_data.zip>.

For some reasons, you need to convert csv files to pickle in the following way. To do this, unzip the archive, move the `.csv` files in this directory and then run the script `csv_to_pkl.py`.

```python
python3 csv_to_pkl.py
```

This makes it possible to use the helpers designed specifically for this data in the Meudon PDR code. For use of this code in other contexts, the data format doesn't really matter.

Generally speaking, `nnbma` handles data via the `RegressionDataset` class, which takes NumPy arrays as arguments. The procedure for loading and pre-processing data is generally what differs most from one use case to another. The learning procedure, on the other hand, is standardized.

## References

[1] Palud, P. & Einig, L. & Le Petit, F. & Bron, E. & Chainais, P. & Chanussot, J. & Pety, J. & Thouvenin, P.-A. & Languignon, D. & Beslić, I. & G. Santa-Maria, M. & Orkisz, J.H. & Ségal, L. & Zakardjian, A. & Bardeau, S. & Gerin, M. & Goicoechea, J.R. & Gratier, P. & Guzman, V. (2023). Neural network-based emulation of interstellar medium models. Astronomy & Astrophysics. 10.1051/0004-6361/202347074.
