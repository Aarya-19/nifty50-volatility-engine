"""
05b_otm_filter.py
─────────────────
Fix: keep only OTM options at each strike.

Root cause of the discontinuity
────────────────────────────────
Both CE and PE existed at the same strikes near the money.
Deep-ITM options are illiquid and their prices produce
unreliable IVs — e.g. a deep-ITM call quoted at intrinsic
value gives a near-zero IV while the corresponding OTM put
correctly shows 13%+.

Fix
───
For each (Date, Expiry) slice:
  - k ≤ 0  →  keep PE  (put is OTM when strike < forward)
  - k > 0  →  keep CE  (call is OTM when strike > forward)

This uses log-moneyness k = log(K/F) computed in 05_moneyness.py,
which is more precise than comparing raw strike vs spot.
"""

import pandas as pd

df = pd.read_csv("surface_data_with_k.csv")

print("Rows before OTM filter:", len(df))

# Keep only the OTM leg at each strike
otm_mask = (
    ((df["k"] <= 0) & (df["Type"] == "PE")) |
    ((df["k"] > 0)  & (df["Type"] == "CE"))
)

df = df[otm_mask].copy()

# Sanity check: no duplicate (Date, Expiry, Strike) should remain
dupes = df.duplicated(subset=["Date", "Expiry", "Strike"]).sum()
assert dupes == 0, f"{dupes} duplicate strikes remain after OTM filter"

print("Rows after  OTM filter:", len(df))
print("Removed:", len(df) - len(df), "ITM / duplicate rows")

df.to_csv("surface_data_otm.csv", index=False)
print("Saved surface_data_otm.csv")

