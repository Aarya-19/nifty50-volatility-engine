import pandas as pd

df = pd.read_csv("options_with_iv.csv")

df = df.dropna(subset=["IV"])

df = df[
    (df["IV"] > 0.03)
    & (df["IV"] < 1.00)
]

print("Remaining rows:", len(df))

df.to_csv(
    "surface_data.csv",
    index=False
)

print("Saved surface_data.csv")
