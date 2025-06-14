
from arch.__future__ import reindexing
from arch.univariate import ConstantMean, EGARCH
from arch.multivariate import DynamicConditionalCorrelation
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("data/ready/master_full.parquet")[["r_BTC","r_SP500"]].dropna()*100

# одиночные EGARCH для маржинальных распределений
mods = []
for col in df:
    cm = ConstantMean(df[col])
    cm.volatility = EGARCH()          # (1,1) по умолчанию
    mods.append(cm.fit(disp="off"))

dcc = DynamicConditionalCorrelation(mods)
dcc_res = dcc.fit(disp="off")

# динамическая корреляция
rho = dcc_res.dynamic_correlations.iloc[:,0]   # BTC-SP500
rho.to_csv("data/ready/dcc_btc_sp500.csv")

plt.figure(figsize=(9,4))
rho.plot()
plt.title("DCC-корреляция BTC ↔ SP500")
plt.tight_layout(); plt.savefig("fig/dcc_btc_sp500.png", dpi=300); plt.close()

