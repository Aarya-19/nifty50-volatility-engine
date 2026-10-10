import pandas as pd
import numpy as np

surface = pd.read_csv("surface_data_otm.csv")
params = pd.read_csv("svi_params.csv")

surface["Date"] = pd.to_datetime(surface["Date"])
surface["Expiry"] = pd.to_datetime(surface["Expiry"])

params["Date"] = pd.to_datetime(params["Date"])
params["Expiry"] = pd.to_datetime(params["Expiry"])

def svi(k, a, b, rho, m, sigma):
    return a + b * (
        rho * (k - m)
        + np.sqrt((k - m) ** 2 + sigma ** 2)
    )

merged = surface.merge(
    params,
    on=["Date", "Expiry"],
    how="inner"
)

merged["T"] = merged["T_x"]

print(merged.columns.tolist())
print(merged.head())

w = svi(
    merged["k"],
    merged["a"],
    merged["b"],
    merged["rho"],
    merged["m"],
    merged["sigma"]
)

merged["IV_SVI"] = np.sqrt(
    np.maximum(w, 0) / merged["T"]
)

merged["Residual"] = (
    merged["IV"] - merged["IV_SVI"]
)

merged.to_csv(
    "richcheap.csv",
    index=False
)
