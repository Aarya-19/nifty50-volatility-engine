import pandas as pd
import numpy as np

r = 0.065

df = pd.read_csv("surface_data.csv")

df["Forward"] = (
    df["Spot"]
    * np.exp(r * df["T"])
)

df["k"] = np.log(
    df["Strike"] /
    df["Forward"]
)

print(df[["Strike","Spot","Forward","k"]].head())

df.to_csv(
    "surface_data_with_k.csv",
    index=False
)

print("Saved surface_data_with_k.csv")
