# garch_step5.py ────────────────────────────────────────────────
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pathlib as pl
from arch.univariate import arch_model

pl.Path("fig").mkdir(exist_ok=True)
pl.Path("data/ready").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/ready/master_full.parquet")
rets = df[["r_BTC", "r_SP500"]].dropna() * 100   # проценты

results = {}
for col in rets:
    am = arch_model(rets[col], vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    results[col] = res
    # полный отчёт
    (pl.Path("data/ready") / f"garch_{col}.txt").write_text(
        res.summary().__str__(), encoding="utf-8"
    )
    # сохраняем условную σ
    res.conditional_volatility.to_csv(
        pl.Path("data/ready") / f"cond_vol_{col}.csv"
    )

# ── общий график двух волатильностей ───────────────────────────
plt.figure(figsize=(10,4))
for col, color in zip(["r_BTC","r_SP500"], ["tab:blue","tab:orange"]):
    sigma = results[col].conditional_volatility
    plt.plot(sigma.index, sigma, color=color, label=col)

plt.title("Условная волатильность GARCH(1,1)")
plt.ylabel("σ_t  (% дневных)")
plt.legend()
plt.tight_layout()
plt.savefig("fig/garch_volatility.png", dpi=300)
plt.close()
print("✓ fig/garch_volatility.png сохранён")

# ── краткий абзац в текстовый файл ─────────────────────────────
mean_btc = results["r_BTC"].conditional_volatility.mean()
mean_sp  = results["r_SP500"].conditional_volatility.mean()
ratio    = mean_btc / mean_sp

summary = f"""
GARCH-анализ (2020-2024)

• Средняя условная дневная σ_BTC  = {mean_btc:.2f} %
• Средняя условная дневная σ_SP500 = {mean_sp:.2f} %
➜ BTC в {ratio:.1f} раз(а) волатильнее фондового индекса.

Пиковые всплески:
• BTC:  {results['r_BTC'].conditional_volatility.idxmax().date()}  σ = {results['r_BTC'].conditional_volatility.max():.2f} %
• SP500:{results['r_SP500'].conditional_volatility.idxmax().date()} σ = {results['r_SP500'].conditional_volatility.max():.2f} %

График: fig/garch_volatility.png  
Полные отчёты: garch_r_BTC.txt, garch_r_SP500.txt
"""

(pl.Path("data/ready") / "garch_summary_for_thesis.txt").write_text(
    summary.strip(), encoding="utf-8"
)
print("✓ garch_summary_for_thesis.txt сохранён")
