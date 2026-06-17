import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/conflicts.csv")

plt.figure(figsize=(8, 5))

plt.hist(df["ttc"], bins=20)

plt.title("TTC Distribution")
plt.xlabel("TTC (seconds)")
plt.ylabel("Frequency")

plt.grid(True)

plt.show()