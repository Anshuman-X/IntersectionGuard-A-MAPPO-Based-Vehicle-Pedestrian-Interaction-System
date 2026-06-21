import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def visualize_pet_distribution(csv_path="results/pet_conflicts.csv", output_image_path="results/pet_distribution.png"):
    """Generate and save a histogram of Post Encroachment Times (PET)."""
    if not os.path.exists(csv_path):
        print(f"Error: Conflict file {csv_path} not found. Cannot generate visualization.")
        return
        
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("No PET conflict records found. Cannot plot histogram.")
        return
        
    # Set plot style parameters for high readability
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'grid.alpha': 0.5
    })
    
    plt.figure(figsize=(8, 5))
    plt.hist(df["PET"], bins=10, range=(0.0, 5.0), color="#5bc0de", edgecolor="black", alpha=0.85, rwidth=0.9)
    plt.title("Distribution of Post Encroachment Time (PET)")
    plt.xlabel("PET (seconds)")
    plt.ylabel("Frequency (Events)")
    plt.xlim(0, 5.0)
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"PET distribution plot successfully saved to {output_image_path}")

if __name__ == "__main__":
    visualize_pet_distribution()
