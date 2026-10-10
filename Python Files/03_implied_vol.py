import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy.optimize import brentq

# Risk-free rate
r = 0.065

df = pd.read_csv("clean_options.csv")


def bs_price(S, K, T, r, sigma, option_type):

    if sigma <= 0:
        return 0

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        return (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    else:
        return (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * norm.cdf(-d1)
        )


def implied_vol(row):

    S = row["Spot"]
    K = row["Strike"]
    T = row["T"]
    price = row["Price"]
    option_type = row["Type"]

    try:

        iv = brentq(
            lambda sigma:
            bs_price(
                S,
                K,
                T,
                r,
                sigma,
                option_type
            )
            - price,
            0.0001,
            5.0
        )

        return iv

    except:
        return np.nan


print("Calculating IVs...")

df["IV"] = df.apply(
    implied_vol,
    axis=1
)

print(df["IV"].describe())

df.to_csv(
    "options_with_iv.csv",
    index=False
)

print("Saved options_with_iv.csv")
