import pandas as pd


def print_conflict_statistics(csv_path="results/conflicts.csv"):
    """Load conflict CSV and print aggregate statistics."""
    df = pd.read_csv(csv_path)

    print("\n===== CONFLICT STATISTICS =====\n")
    print("Total Conflicts :", len(df))
    print("Average TTC     :", round(df["ttc"].mean(), 2))
    print("Minimum TTC     :", round(df["ttc"].min(), 2))
    print("Maximum TTC     :", round(df["ttc"].max(), 2))
    print("Average Speed   :", round(df["speed"].mean(), 2))
    print("Maximum Speed   :", round(df["speed"].max(), 2))
    print("\n==============================")


if __name__ == "__main__":
    print_conflict_statistics()