import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

# 0. Создаём директорию для результатов
os.makedirs("results", exist_ok=True)

# 1. Загрузка и подготовка данных
df = pd.read_parquet("data/ready/master_full.parquet").sort_index()

# 1.1. Масштабируем usdt_supply в млрд и считаем дневной прирост
df["usdt_b"]   = df["usdt_supply"] / 1e9
df["dUSDT_b"]  = df["usdt_b"].diff()

# 1.2. Целевая - дневное изменение ставки CPF3M
df["dCPF3M"]   = df["CPF3M"].diff()

# 1.3. Убираем пропуски перед созданием лагов
df = df.dropna(subset=["dUSDT_b", "DFF", "dCPF3M"])

# 2. Создаём инструменты: лаги прироста 1…7 дней
for lag in range(1, 8):
    df[f"z{lag}"] = df["dUSDT_b"].shift(lag)
df2 = df.dropna(subset=[f"z{lag}" for lag in range(1, 8)])

# 3. Формируем матрицы для IV 2SLS
Y     = df2["dCPF3M"]                          # ΔCPF3M, %-пункты
endog = df2["dUSDT_b"]                         # прирост USDT, млрд USD
exog  = sm.add_constant(df2[["DFF"]])          # константа + overnight rate
instr = df2[[f"z{lag}" for lag in range(1, 8)]] # лаги инструмента

# 4. Оценка IV 2SLS
iv = IV2SLS(Y, exog, endog, instr).fit(cov_type="robust")

# 5. Сохраняем сводку регрессии в текстовый файл
with open("results/iv_regression_summary.txt", "w", encoding="utf-8") as out:
    out.write(iv.summary.as_text())

# 6. Вычисляем эффект шока +5 млрд USDT и дописываем в файл
beta_iv    = iv.params["dUSDT_b"]
shock_bn   = 5.0  # млрд USD
effect_pct = beta_iv * shock_bn       # изменение в %-пунктах
effect_bp  = effect_pct * 100         # изменение в базисных пунктах

with open("results/iv_regression_summary.txt", "a", encoding="utf-8") as out:
    out.write(f"\n\nShock +{shock_bn:.0f} млрд USDT ⇒ ΔCPF3M ≈ {effect_bp:.2f} б.п.\n")

# 7. Строим и сохраняем график влияния шока на CPF3M в середине периода
# выбираем точку шока как середину индекса df2
mid_idx     = len(df2) // 2
shock_date  = df2.index[mid_idx]
cpf3m_series = df2["CPF3M"]
cpf3m_shock  = cpf3m_series.copy()
# применяем шок, начиная с середины
cpf3m_shock.loc[shock_date:] += effect_pct

plt.figure(figsize=(10, 5))
plt.plot(cpf3m_series.index, cpf3m_series, label="CPF3M (факт)")
plt.plot(cpf3m_shock.index, cpf3m_shock, "--", label=f"CPF3M + шок {shock_bn:.0f} млрд")
plt.axvline(shock_date, color="red", linestyle=":", label=f"дата шока {shock_date.date()}")
plt.title("Влияние шока эмиссии +5 млрд USDT на ставку CPF3M")
plt.ylabel("CPF3M, % годовых")
plt.legend()
plt.tight_layout()
plt.savefig("results/iv_shock_plot.png", dpi=150)
plt.show()
