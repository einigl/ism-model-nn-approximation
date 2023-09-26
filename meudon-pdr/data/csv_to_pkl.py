import os

import pandas as pd

empty = True
for file in os.listdir():
    if file.endswith(".csv"):
        empty = False
        print(file, end="\t", flush=True)
        df = pd.read_csv(file)
        df.to_pickle(file.replace(".csv", ".pkl"))
        print("Done")

if empty:
    print("There is no .csv file in this directory. Maybe you need to move files from subfolders to the current directory.")
