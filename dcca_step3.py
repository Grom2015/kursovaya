# dcca_step3.py
import pandas as pd, numpy as np
from dcca_utils import dcca_coefficient
import matplotlib.pyplot as plt

df = pd.read_parquet("data/ready/master_full.parquet")

pairs = [("r_BTC","r_SP500"),
         ("r_BTC","r_GOLD"),
         ("r_BTC","r_CPF3M")]  # создадим r_CPF3M = log(CPF3M/shift)

df["r_CPF3M"] = np.log(df["CPF3M"] / df["CPF3M"].shift(1))

lags = [32, 64, 128, 256]
results = {}

for x, y in pairs:
    series_x, series_y = df[x].dropna(), df[y].dropna()
    joint = series_x.align(series_y, join="inner")
    res = [dcca_coefficient(joint[0].values, joint[1].values, lag=L) for L in lags]
    results[f"{x}-{y}"] = res

# таблица результата
dcca_out = pd.DataFrame(results, index=lags)
dcca_out.to_csv("data/ready/dcca_results.csv")
print(dcca_out)
