import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "surface_data_otm.csv"
)

date = "2024-09-02"

sample = df[
    df["Date"] == date
]

expiry = sorted(
    sample["Expiry"].unique()
)[0]

sample = sample[
    sample["Expiry"] == expiry
]

# ADD THIS HERE
sample = sample[
    sample["k"].between(-0.15, 0.15)
]

print("Date:", date)
print("Expiry:", expiry)
print("Contracts:", len(sample))

plt.scatter(
    sample["Strike"],
    sample["IV"]
)

plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title(
    f"{date} | {expiry}"
)

plt.savefig(
    "vol_smile.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved vol_smile.png")
