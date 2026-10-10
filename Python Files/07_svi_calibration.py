import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

df = pd.read_csv(
    "surface_data_otm.csv"
)

# Same smile used in 06_vol_smile.py

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

sample = sample[
    sample["k"].between(-0.15, 0.15)
]

# Total variance
sample["w"] = (
    sample["IV"]**2
) * sample["T"]

# SVI function

def svi(
    k,
    a,
    b,
    rho,
    m,
    sigma
):

    return (
        a
        + b * (
            rho * (k - m)
            + np.sqrt(
                (k - m)**2
                + sigma**2
            )
        )
    )

params, _ = curve_fit(
    svi,
    sample["k"],
    sample["w"],
    p0=[
        0.01,
        0.1,
        0.0,
        0.0,
        0.1
    ],
    maxfev=100000
)

a,b,rho,m,sigma = params

print("\nSVI PARAMETERS")
print("----------------")
print("a =", a)
print("b =", b)
print("rho =", rho)
print("m =", m)
print("sigma =", sigma)

# Plot

k_grid = np.linspace(
    sample["k"].min(),
    sample["k"].max(),
    200
)

plt.scatter(
    sample["k"],
    sample["w"],
    label="Market"
)

plt.plot(
    k_grid,
    svi(
        k_grid,
        a,b,rho,m,sigma
    ),
    label="SVI Fit"
)

plt.xlabel("k")
plt.ylabel("Total Variance")

plt.legend()

plt.savefig(
    "svi_fit.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved svi_fit.png")
