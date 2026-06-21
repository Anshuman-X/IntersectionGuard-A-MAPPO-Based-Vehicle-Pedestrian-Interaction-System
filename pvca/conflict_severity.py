import pandas as pd


def print_conflict_severity(csv_path="results/conflicts.csv"):
    """Categorize conflicts by TTC severity bands."""
    df = pd.read_csv(csv_path)

    critical = len(df[df["ttc"] < 1])
    moderate = len(df[(df["ttc"] >= 1) & (df["ttc"] < 2)])
    low = len(df[(df["ttc"] >= 2) & (df["ttc"] < 3)])

    print("\n===== CONFLICT SEVERITY =====\n")
    print("Critical (<1s) :", critical)
    print("Moderate (1-2s):", moderate)
    print("Low (2-3s)     :", low)
    print("\n============================")


if __name__ == "__main__":
    print_conflict_severity()