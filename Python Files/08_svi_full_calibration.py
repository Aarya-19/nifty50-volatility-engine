import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv("surface_data_otm.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Expiry"] = pd.to_datetime(df["Expiry"])

results = []

def svi(k,a,b,rho,m,sigma):
    return a + b*(
        rho*(k-m)
        + np.sqrt((k-m)**2 + sigma**2)
    )

for (date, expiry), g in df.groupby(["Date","Expiry"]):

    if len(g) < 10:
        continue

    k = g["k"].values

    T = g["T"].iloc[0]

    w_market = (g["IV"].values**2) * T

    def objective(params):
        a,b,rho,m,sigma = params

        if b <= 0:
            return 1e10

        if sigma <= 0:
            return 1e10

        if abs(rho) >= 1:
            return 1e10

        w_model = svi(k,a,b,rho,m,sigma)

        return np.mean((w_market-w_model)**2)

    x0 = [
        0.001,
        0.01,
        -0.5,
        0.0,
        0.05
    ]

    try:

        res = minimize(
            objective,
            x0,
            method="Nelder-Mead"
        )

        a,b,rho,m,sigma = res.x

        results.append({
            "Date":date,
            "Expiry":expiry,
            "T":T,
            "a":a,
            "b":b,
            "rho":rho,
            "m":m,
            "sigma":sigma,
            "n":len(g)
        })

    except:
        pass

params = pd.DataFrame(results)

params.to_csv(
    "svi_params.csv",
    index=False
)

print(params.head())
print("\nSlices fitted:",len(params))
