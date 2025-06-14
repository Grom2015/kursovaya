
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import pathlib as pl

pl.Path("fig").mkdir(exist_ok=True)

df = pd.read_parquet("data/ready/master_full.parquet")

# ── 2.1 Корреляционная матрица доходностей ─────────────────────
ret_cols = [c for c in df.columns if c.startswith("r_")]
corr = df[ret_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Матрица корреляций лог-доходностей (2020-2024)")
plt.tight_layout()
plt.savefig("fig/heatmap_returns.png", dpi=300)
plt.close()
print("✓ heatmap_returns.png")

# ── 2.2 Распределения доходностей BTC vs SP500 ─────────────────
for col in ["r_BTC", "r_SP500"]:
    sns.histplot(df[col].dropna(), kde=True, stat="density", bins=50)
    plt.title(f"Распределение {col}")
    plt.tight_layout()
    plt.savefig(f"fig/hist_{col}.png", dpi=300)
    plt.close()
print("✓ histograms r_BTC / r_SP500 сохранены")

# ── 2.3 Скользящее среднее и волатильность BTC ─────────────────
btc = df["r_BTC"].dropna()
roll_mean = btc.rolling(30).mean()
roll_std  = btc.rolling(30).std()

plt.figure(figsize=(9,4))
plt.plot(btc.index, btc, label="r_BTC", alpha=.4)
plt.plot(roll_mean.index, roll_mean, label="30-day mean")
plt.plot(roll_std.index, roll_std, label="30-day stdev")
plt.legend(); plt.title("BTC: лог-доходность, скользящее ср. и σ")
plt.tight_layout(); plt.savefig("fig/btc_roll.png", dpi=300); plt.close()
print("✓ btc_roll.png")

# ── 2.4 Самкорреляция прироста USDT ────────────────────────────
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(6,3))
plot_acf(df["dUSDT"].dropna(), lags=30, zero=False)
plt.title("ACF ΔUSDT (30 лагов)")
plt.tight_layout(); plt.savefig("fig/acf_dUSDT.png", dpi=300); plt.close()
print("✓ acf_dUSDT.png")

# ── 2.5 Таблица описательной статистики доходностей ────────────
desc_ret = df[ret_cols].describe().T.round(4)
desc_ret.to_csv("data/ready/desc_returns.csv")
print("✓ desc_returns.csv")
