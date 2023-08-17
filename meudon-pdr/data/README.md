This directory should contains data downloaded from https://ism.obspm.fr/files/ArticleData/2023_Palud_Einig/2023_Palud_Einig_data.zip.

For some reasons, you need to convert csv files to pickle in the following way. To do this, unzip the archive, move the `.csv` files in this directory and then run the script `csv_to_pkl.py`.

```python
python3 csv_to_pkl.py
```

This makes it possible to use the helpers designed specifically for this data in the Meudon PDR code. For use of this code in other contexts, the data format doesn't really matter.

Generally speaking, `nnbma` handles data via the `RegressionDataset` class, which takes NumPy arrays as arguments. The procedure for loading and pre-processing data is generally what differs most from one use case to another. The learning procedure, on the other hand, is standardized.
