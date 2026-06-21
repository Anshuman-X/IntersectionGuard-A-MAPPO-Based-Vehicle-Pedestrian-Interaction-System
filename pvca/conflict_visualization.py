import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_conflict_histogram(csv_path="results/conflicts.csv", output_path="results/conflict_ttc_histogram.png"):
    """Plot TTC distribution histogram from conflict data."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run the conflict detector first.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No conflicts to visualize.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df["ttc"], bins=20, color="#d9534f", edgecolor="black", alpha=0.85, rwidth=0.9)
    plt.title("TTC Distribution — Vehicle Conflicts")
    plt.xlabel("TTC (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    plot_conflict_histogram()