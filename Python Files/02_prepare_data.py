import pandas as pd

# Load data
df = pd.read_csv("Nifty_Options_Historical_2022_2025.csv")

# Convert dates
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d/%m/%y"
)

df["Expiry"] = pd.to_datetime(
    df["Expiry"],
    format="%d/%m/%y"
)

# Time to expiry
df["T"] = (
    df["Expiry"] - df["Date"]
).dt.days / 365

print("Rows before filtering:", len(df))

# Step 2: Filter bad observations
df = df[df["Price"] > 1]
df = df[df["T"] > 0]

print("Rows after filtering:", len(df))

print(df.head())
print(df["Type"].value_counts())

df.to_csv(
    "clean_options.csv",
    index=False
)

print("Saved clean_options.csv")
