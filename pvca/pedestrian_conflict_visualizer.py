import pandas as pd
import matplotlib.pyplot as plt
import os

# Set plotting style for academic publication
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'legend.fontsize': 11
})

def visualize_pedestrian_conflicts(csv_path="results/pedestrian_conflicts.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Fill NaN values in 'zone' with 'General (None)'
    df["zone"] = df["zone"].fillna("General (None)")
    
    # Create output directory for figures if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # --- Figure 1: TTC Distribution ---
    ttc_df = df[df["conflict_type"].isin(["TTC", "Overlap"])]
    if not ttc_df.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(ttc_df["value"], bins=25, range=(0, 3.0), color="#d9534f", edgecolor="black", alpha=0.85, rwidth=0.9)
        plt.title("Distribution of Vehicle-Pedestrian Time-to-Collision (TTC)")
        plt.xlabel("TTC (seconds)")
        plt.ylabel("Frequency (Steps)")
        plt.xlim(0, 3.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("results/pedestrian_ttc_distribution.png", dpi=300)
        plt.close()
        print("Saved: results/pedestrian_ttc_distribution.png")
        
    # --- Figure 2: PET Distribution ---
    pet_df = df[df["conflict_type"] == "PET"]
    if not pet_df.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(pet_df["value"], bins=10, range=(0, 5.0), color="#5bc0de", edgecolor="black", alpha=0.85, rwidth=0.9)
        plt.title("Distribution of Post Encroachment Time (PET)")
        plt.xlabel("PET (seconds)")
        plt.ylabel("Frequency (Events)")
        plt.xlim(0, 5.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("results/pedestrian_pet_distribution.png", dpi=300)
        plt.close()
        print("Saved: results/pedestrian_pet_distribution.png")
        
    # --- Figure 3: Severity Breakdown by Conflict Type ---
    # Prepare data for grouped bar chart
    # Ensure categories are in specific order: Critical, Moderate, Low
    severity_order = ['Critical', 'Moderate', 'Low']
    
    # Reindex to ensure all categories exist
    severity_by_type = df.groupby(["conflict_type", "severity"]).size().unstack(fill_value=0)
    
    # Reorder columns
    existing_cols = [col for col in severity_order if col in severity_by_type.columns]
    severity_by_type = severity_by_type[existing_cols]
    
    # Custom color palette mapping
    color_map = {"Critical": "#d9534f", "Moderate": "#f0ad4e", "Low": "#5bc0de"}
    colors = [color_map[col] for col in severity_by_type.columns]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    severity_by_type.plot(kind="bar", ax=ax, color=colors, edgecolor="black", alpha=0.85, width=0.7)
    plt.title("Conflict Severity Breakdown by Type")
    plt.xlabel("Conflict Type")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Severity Level")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig("results/pedestrian_conflict_severity.png", dpi=300)
    plt.close()
    print("Saved: results/pedestrian_conflict_severity.png")
    
    # --- Figure 4: Spatial Breakdown ---
    plt.figure(figsize=(8, 5))
    zone_counts = df["zone"].value_counts()
    
    # Set nice colors for bars
    bar_colors = ["#428bca", "#5cb85c", "#f0ad4e"][:len(zone_counts)]
    
    plt.bar(zone_counts.index, zone_counts.values, color=bar_colors, edgecolor="black", alpha=0.85, width=0.5)
    plt.title("Pedestrian-Vehicle Conflicts by Intersection Zone")
    plt.xlabel("Crossing / Conflict Zone")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig("results/pedestrian_conflict_zones.png", dpi=300)
    plt.close()
    print("Saved: results/pedestrian_conflict_zones.png")
    
    print("\nVisualization complete! All figures saved in the 'results/' folder.")

if __name__ == "__main__":
    visualize_pedestrian_conflicts()
