import pandas as pd
import numpy as np

input_path = "invest-sim/data/50ETF/Filtered_OptionInstruments_510050.pkl"
output_path = "invest-sim/data/50ETF/Filtered_OptionInstruments_FIXED.pkl"

print("Loading instruments file...")
df = pd.read_pickle(input_path)

print("Original shape:", df.shape)
print("Original strike_price sample:", df["strike_price"].iloc[0])

# ---- 核心修复：explode strike_price ----

# 1) explode（将 list/ndarray 拆成多行）
df = df.explode("strike_price")

# 2) 转 float
df["strike_price"] = df["strike_price"].astype(float)

# 3) 重新排序（可选）
df = df.sort_values(["maturity_date", "option_type", "strike_price"]).reset_index(drop=True)

print("Fixed shape:", df.shape)
print("Fixed strike_price sample:", df["strike_price"].head())

print("Saving to:", output_path)
df.to_pickle(output_path)

print("DONE. New instrument file created successfully.")
