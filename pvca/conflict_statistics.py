import pandas as pd

# Load conflict data
df = pd.read_csv("results/conflicts.csv")

print("\n===== CONFLICT STATISTICS =====\n")

print("Total Conflicts :", len(df))

print("Average TTC     :", round(df["ttc"].mean(), 2))

print("Minimum TTC     :", round(df["ttc"].min(), 2))

print("Maximum TTC     :", round(df["ttc"].max(), 2))

print("Average Speed   :", round(df["speed"].mean(), 2))

print("Maximum Speed   :", round(df["speed"].max(), 2))

print("\n==============================")