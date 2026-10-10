"""
13_metrics.py
─────────────
Computes Sharpe ratio, max drawdown, and walk-forward
performance on NORMALISED daily returns (% of capital deployed).

Sharpe is computed on daily portfolio returns, where:
  daily_return = sum(trade P&L on day t) / sum(notional on day t)

This gives a capital-normalised Sharpe that is directly
comparable to industry benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SPLIT_DATE   = "2024-10-01"
ANNUAL_SCALE = 252

# ── Load trades ───────────────────────────────────────────────────────────────
trades = pd.read_csv("trades.csv")
trades["Date"] = pd.to_datetime(trades["Date"])

# ── Daily return = total P&L / total notional deployed that day ───────────────
daily = (
    trades.groupby(["Date", "Split"])
    .agg(
        Daily_PnL      = ("Total_PnL", "sum"),
        Daily_Notional = ("Notional",  "sum")
    )
    .reset_index()
    .sort_values("Date")
)

daily["Daily_Return"] = daily["Daily_PnL"] / daily["Daily_Notional"]

daily_is  = daily[daily["Split"] == "IS" ]["Daily_Return"].values
daily_oos = daily[daily["Split"] == "OOS"]["Daily_Return"].values

# ── Metric helpers ────────────────────────────────────────────────────────────
def sharpe(r_arr):
    if len(r_arr) < 2 or r_arr.std() == 0:
        return np.nan
    return (r_arr.mean() / r_arr.std()) * np.sqrt(ANNUAL_SCALE)

def max_drawdown_pct(r_arr):
    cum = np.cumprod(1 + r_arr)
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / roll_max
    return dd.min() * 100   # as percentage

def win_rate(r_arr):
    return (r_arr > 0).mean()

# ── Calculate ─────────────────────────────────────────────────────────────────
metrics = {}
for label, arr in [("In-Sample (IS)", daily_is), ("Out-of-Sample (OOS)", daily_oos)]:
    metrics[label] = {
        "Trading Days"        : len(arr),
        "Sharpe Ratio"        : round(sharpe(arr), 2),
        "Max Drawdown (%)"    : round(max_drawdown_pct(arr), 2),
        "Win Rate"            : f"{win_rate(arr)*100:.1f}%",
        "Avg Daily Return (%)" : f"{arr.mean()*100:.4f}%",
        "Total Return (%)"    : f"{np.sum(arr)*100:.2f}%",
    }

# ── Print ─────────────────────────────────────────────────────────────────────
print("=" * 58)
print("      WALK-FORWARD PERFORMANCE (Normalised Returns)")
print("=" * 58)
print(f"  Normalisation: P&L per lot / (Strike × Lot size)")
print(f"  Lot size: 25  |  Split: {SPLIT_DATE}")
print()

for period, m in metrics.items():
    print(f"  {period}")
    print(f"  {'─'*44}")
    for k, v in m.items():
        print(f"    {k:<26}: {v}")
    print()

is_sharpe  = metrics["In-Sample (IS)"]["Sharpe Ratio"]
oos_sharpe = metrics["Out-of-Sample (OOS)"]["Sharpe Ratio"]
decay = (is_sharpe - oos_sharpe) / abs(is_sharpe) * 100 if is_sharpe != 0 else 0

print("=" * 58)
print(f"  IS → OOS Sharpe decay : {decay:.1f}%")
if decay < 30:
    print("  ✓ Low decay — strategy generalises well")
elif decay < 60:
    print("  ~ Moderate decay — typical for vol strategies")
else:
    print("  ✗ High decay — consider adjusting zscore threshold")
print("=" * 58)

# ── Save ──────────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv("metrics_summary.csv")
print("\nSaved metrics_summary.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
all_returns = daily["Daily_Return"].values
all_dates   = daily["Date"].values
cum_returns = np.cumprod(1 + all_returns) - 1   # cumulative % return

split_ts  = pd.Timestamp(SPLIT_DATE)
split_idx = np.searchsorted(all_dates, split_ts)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    all_dates[:split_idx],
    cum_returns[:split_idx] * 100,
    color="#1f77b4", linewidth=1.8, label="In-Sample"
)
ax.plot(
    all_dates[split_idx:],
    cum_returns[split_idx:] * 100,
    color="#ff7f0e", linewidth=1.8, label="Out-of-Sample"
)
ax.axvline(split_ts, color="gray", linestyle="--", linewidth=1.2,
           label=f"Walk-Forward Split ({SPLIT_DATE})")
ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)

ax.set_title(
    "Nifty 50 Delta-Neutral Vol Strategy — Cumulative Return (% of Notional)",
    fontsize=13
)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (%)")
ax.legend()
ax.grid(alpha=0.3)

ax.text(0.02, 0.95, f"IS  Sharpe: {is_sharpe}",
        transform=ax.transAxes, fontsize=10,
        color="#1f77b4", verticalalignment="top")
ax.text(0.02, 0.88, f"OOS Sharpe: {oos_sharpe}",
        transform=ax.transAxes, fontsize=10,
        color="#ff7f0e", verticalalignment="top")

plt.tight_layout()
plt.savefig("pnl_curve.png", dpi=300, bbox_inches="tight")
print("Saved pnl_curve.png")
plt.show()
