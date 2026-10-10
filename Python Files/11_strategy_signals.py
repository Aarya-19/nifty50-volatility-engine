import pandas as pd

df = pd.read_csv("richcheap.csv")

# Z-score residuals within each date-expiry slice

df["zscore"] = (
    df.groupby(["Date", "Expiry"])["Residual"]
    .transform(
        lambda x: (x - x.mean()) / x.std()
    )
)

df["Signal"] = "NONE"

df.loc[
    df["zscore"] > 2,
    "Signal"
] = "SELL_VOL"

df.loc[
    df["zscore"] < -2,
    "Signal"
] = "BUY_VOL"

df.to_csv(
    "signals.csv",
    index=False
)

print("\nSignal Counts")
print(df["Signal"].value_counts())

print("\nSample Signals")
print(
    df[df["Signal"] != "NONE"][
        [
            "Date",
            "Expiry",
            "Strike",
            "Type",
            "Residual",
            "zscore",
            "Signal"
        ]
    ].head(20)
)
