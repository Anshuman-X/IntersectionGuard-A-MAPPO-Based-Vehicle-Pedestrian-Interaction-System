import pandas as pd

df = pd.read_csv("results/conflicts.csv")

critical = len(df[df["ttc"] < 1])

moderate = len(
    df[(df["ttc"] >= 1) &
       (df["ttc"] < 2)]
)

low = len(
    df[(df["ttc"] >= 2) &
       (df["ttc"] < 3)]
)

print("\n===== CONFLICT SEVERITY =====\n")

print("Critical (<1s) :", critical)
print("Moderate (1-2s):", moderate)
print("Low (2-3s)     :", low)

print("\n============================")