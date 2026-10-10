"""
12_backtest.py
──────────────
Delta-neutral long-cheap / short-rich vol backtest.

Strategy Logic
──────────────
For each signal (BUY_VOL or SELL_VOL) on day t:
  - Enter at close of day t
  - Exit  at close of day t+1
  - Delta-hedge the option position using the underlying (Spot)

P&L Decomposition (vega-based, standard desk approach)
──────────────────────────────────────────────────────
  Option P&L  ≈ direction * vega * (IV_exit - IV_entry) * LOT_SIZE
  Hedge  P&L  = -direction * delta * (Spot_exit - Spot_entry) * LOT_SIZE
  Transaction cost = 0.05% of option price per leg (entry + exit) * LOT_SIZE

Normalisation
─────────────
  Each trade is sized at 1 lot (Nifty lot size = 25 units post-2024).
  Capital per trade = Strike * LOT_SIZE (option notional).
  Daily return = daily P&L / total capital deployed that day.
  Sharpe is computed on these daily returns (annualised x sqrt(252)).

Walk-Forward Split
──────────────────
  In-sample  (IS) : 2024-06-21 → 2024-09-30
  Out-of-sample (OOS): 2024-10-01 → 2024-12-31
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# ── Constants ────────────────────────────────────────────────────────────────
r              = 0.065
TRANSACTION_BP = 0.0005      # 5 bps per leg
SPLIT_DATE     = "2024-10-01"
LOT_SIZE       = 25          # Nifty 50 lot size (revised in 2024)

# ── Load signals ─────────────────────────────────────────────────────────────
df = pd.read_csv("signals.csv")
df["Date"]   = pd.to_datetime(df["Date"])
df["Expiry"] = pd.to_datetime(df["Expiry"])

signals = df[df["Signal"] != "NONE"].copy()
signals = signals.sort_values("Date").reset_index(drop=True)

# ── Build next-day lookup ─────────────────────────────────────────────────────
all_dates   = sorted(df["Date"].unique())
date_to_next = {
    d: all_dates[i + 1]
    for i, d in enumerate(all_dates[:-1])
}

df_indexed = df.set_index(["Date", "Expiry", "Strike", "Type"])

# ── Black-Scholes helpers ─────────────────────────────────────────────────────
def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_delta(S, K, T, r, sigma, option_type):
    d1 = bs_d1(S, K, T, r, sigma)
    if np.isnan(d1):
        return np.nan
    return norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1.0

def bs_vega(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    if np.isnan(d1):
        return np.nan
    return S * np.sqrt(T) * norm.pdf(d1)

# ── Backtest loop ─────────────────────────────────────────────────────────────
records = []

for _, row in signals.iterrows():

    date_t   = row["Date"]
    expiry   = row["Expiry"]
    strike   = row["Strike"]
    opt_type = row["Type"]
    signal   = row["Signal"]

    if date_t not in date_to_next:
        continue
    date_t1 = date_to_next[date_t]

    IV_entry    = row["IV"]
    T_entry     = row["T"]
    price_entry = row["Price"]
    spot_entry  = row["Spot"]

    try:
        exit_row = df_indexed.loc[(date_t1, expiry, strike, opt_type)]
        if isinstance(exit_row, pd.DataFrame):
            exit_row = exit_row.iloc[0]
        IV_exit   = exit_row["IV"]
        spot_exit = exit_row["Spot"]
    except KeyError:
        continue

    if np.isnan(IV_entry) or np.isnan(IV_exit):
        continue

    direction = 1.0 if signal == "BUY_VOL" else -1.0

    delta = bs_delta(spot_entry, strike, T_entry, r, IV_entry, opt_type)
    vega  = bs_vega(spot_entry,  strike, T_entry, r, IV_entry)

    if np.isnan(delta) or np.isnan(vega):
        continue

    # Per-lot P&L
    option_pnl = direction * vega  * (IV_exit - IV_entry) * LOT_SIZE
    hedge_pnl  = -direction * delta * (spot_exit - spot_entry) * LOT_SIZE
    txn_cost   = 2 * TRANSACTION_BP * price_entry * LOT_SIZE

    total_pnl  = option_pnl + hedge_pnl - txn_cost

    # Notional capital for this trade (option notional = strike * lot)
    notional   = strike * LOT_SIZE

    records.append({
        "Date"       : date_t,
        "Expiry"     : expiry,
        "Strike"     : strike,
        "Type"       : opt_type,
        "Signal"     : signal,
        "IV_entry"   : IV_entry,
        "IV_exit"    : IV_exit,
        "Delta"      : delta,
        "Vega"       : vega,
        "Option_PnL" : option_pnl,
        "Hedge_PnL"  : hedge_pnl,
        "Txn_Cost"   : txn_cost,
        "Total_PnL"  : total_pnl,
        "Notional"   : notional,
        "Return_pct" : total_pnl / notional,   # per-trade return
        "Split"      : "IS" if date_t < pd.Timestamp(SPLIT_DATE) else "OOS"
    })

trades = pd.DataFrame(records)
trades.to_csv("trades.csv", index=False)

print(f"Total trades executed : {len(trades)}")
print(f"  In-sample  (IS)     : {(trades['Split']=='IS').sum()}")
print(f"  Out-of-sample (OOS) : {(trades['Split']=='OOS').sum()}")
print(f"\nLot size used         : {LOT_SIZE}")
print(f"Mean per-trade return : {trades['Return_pct'].mean()*100:.4f}%")
print("\nSaved trades.csv")
