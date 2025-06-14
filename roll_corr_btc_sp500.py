
import pandas as pd, matplotlib.pyplot as plt, pathlib as pl

df = pd.read_parquet("data/ready/master_full.parquet")[["r_BTC","r_SP500"]].dropna()

window = 128
rho_roll = df["r_BTC"].rolling(window).corr(df["r_SP500"])

pl.Path("fig").mkdir(exist_ok=True)

plt.figure(figsize=(9,4))
rho_roll.plot()
plt.axhline(0, color="gray", lw=0.8)
plt.title(f"{window}-дневная скользящая корреляция BTC ↔ SP500")
plt.ylabel("ρ")
plt.tight_layout()
plt.savefig("fig/roll_corr_btc_sp500.png", dpi=300)
plt.close()


