import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("surface_data_otm.csv")

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df["k"],
    df["T"],
    df["IV"],
    s=3
)

ax.set_xlabel("k")
ax.set_ylabel("T")
ax.set_zlabel("IV")

plt.savefig(
    "market_surface.png",
    dpi=300
)

plt.show()
