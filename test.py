import sys
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "invest-sim"))

from invest_sim.options.data import OptionDataStore

store = OptionDataStore(
    instruments_path="invest-sim/data/50ETF/sample/Filtered_OptionInstruments_sample.pkl",
    prices_path="invest-sim/data/50ETF/sample/Filtered_OptionPrice_sample.feather",
)
store.load()

# Vectorized validation — no per-ID get_price_history calls
replayable = store._replayable_instruments.copy()
print(f"Replayable instruments count: {len(replayable)}")

prices = store._prices.copy()
if "close" not in prices.columns:
    raise RuntimeError("Price file missing 'close' column")

prices["obid_str"] = prices["order_book_id"].astype(str)
close_counts = prices[prices["close"].notna()].groupby("obid_str").size()
row_counts = prices.groupby("obid_str").size()

replayable_ids = [str(x) for x in store._replayable_ids]

bad_rows = []
good_rows = []

for _, row in replayable.iterrows():
    obid = str(row["order_book_id"])
    symbol = row["symbol"]
    expiry = row["maturity_date"]
    strike = row["strike_price"]
    opt_type = row["option_type"]

    total_rows = row_counts.get(obid, 0)
    valid_close = close_counts.get(obid, 0)

    if total_rows == 0:
        bad_rows.append((symbol, obid, expiry, strike, opt_type, "EMPTY PRICE DF"))
        continue
    if valid_close < 2:
        bad_rows.append((symbol, obid, expiry, strike, opt_type, f"NOT ENOUGH CLOSE DATA ({valid_close})"))
        continue

    good_rows.append((symbol, obid, expiry, strike, opt_type, total_rows, valid_close))

print("\n=== Verification Summary ===")
print(f"Total replayable instruments: {len(replayable)}")
print(f"Valid replayable instruments: {len(good_rows)}")
print(f"Invalid replayable instruments (should be ZERO): {len(bad_rows)}")

if bad_rows:
    print("\n⚠️ These replayable instruments actually CANNOT replay (should be removed):")
    bad_df = pd.DataFrame(
        bad_rows,
        columns=["symbol", "order_book_id", "expiry", "strike", "type", "issue"],
    )
    print(bad_df)
else:
    print("\n🎉 All replayable instruments have valid price history — System is consistent!")
