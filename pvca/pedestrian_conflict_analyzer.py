import pandas as pd
import os

def analyze_pedestrian_conflicts(csv_path="results/pedestrian_conflicts.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*50)
    print("      PEDESTRIAN-VEHICLE CONFLICT ANALYSIS REPORT")
    print("="*50 + "\n")
    
    # 1. Overview
    total_records = len(df)
    print(f"Total Conflict Records: {total_records}")
    
    # Counts by conflict type
    type_counts = df["conflict_type"].value_counts()
    print("\n--- Breakdown by Conflict Type ---")
    for c_type, count in type_counts.items():
        percentage = (count / total_records) * 100
        print(f"  * {c_type:10} : {count:5} ({percentage:.1f}%)")
        
    # 2. Severity analysis
    print("\n--- Breakdown by Severity ---")
    severity_counts = df["severity"].value_counts()
    for sev, count in severity_counts.items():
        percentage = (count / total_records) * 100
        print(f"  * {sev:10} : {count:5} ({percentage:.1f}%)")
        
    # Severity within each conflict type
    print("\n--- Severity by Conflict Type ---")
    grouped_severity = df.groupby(["conflict_type", "severity"]).size().unstack(fill_value=0)
    print(grouped_severity.to_string())
    
    # 3. Spatial breakdown
    print("\n--- Breakdown by Conflict Zone (Crosswalk) ---")
    zone_counts = df["zone"].value_counts()
    for zone, count in zone_counts.items():
        percentage = (count / total_records) * 100
        print(f"  * {zone:10} : {count:5} ({percentage:.1f}%)")
        
    # 4. Detailed metrics for TTC (TTC and Overlap)
    ttc_df = df[df["conflict_type"].isin(["TTC", "Overlap"])]
    if not ttc_df.empty:
        print("\n--- Time-to-Collision (TTC) Metrics ---")
        print(f"  * Average TTC : {ttc_df['value'].mean():.2f} s")
        print(f"  * Minimum TTC : {ttc_df['value'].min():.2f} s")
        print(f"  * Maximum TTC : {ttc_df['value'].max():.2f} s")
        print(f"  * Average Dist: {ttc_df['distance'].mean():.2f} m")
        print(f"  * Minimum Dist: {ttc_df['distance'].min():.2f} m")
        
    # 5. Detailed metrics for PET
    pet_df = df[df["conflict_type"] == "PET"]
    if not pet_df.empty:
        print("\n--- Post Encroachment Time (PET) Metrics ---")
        print(f"  * Average PET : {pet_df['value'].mean():.2f} s")
        print(f"  * Minimum PET : {pet_df['value'].min():.2f} s")
        print(f"  * Maximum PET : {pet_df['value'].max():.2f} s")
        
    # 6. Speed characteristics
    print("\n--- Speed Characteristics during Conflicts ---")
    print(f"  * Average Vehicle Speed    : {df['vehicle_speed'].mean():.2f} m/s ({df['vehicle_speed'].mean()*3.6:.1f} km/h)")
    print(f"  * Maximum Vehicle Speed    : {df['vehicle_speed'].max():.2f} m/s ({df['vehicle_speed'].max()*3.6:.1f} km/h)")
    print(f"  * Average Pedestrian Speed : {df['pedestrian_speed'].mean():.2f} m/s ({df['pedestrian_speed'].mean()*3.6:.1f} km/h)")
    
    # Save a summary statistics file
    summary_path = "results/pedestrian_conflict_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PEDESTRIAN-VEHICLE CONFLICT ANALYSIS SUMMARY\n")
        f.write("="*45 + "\n\n")
        f.write(f"Total Conflicts: {total_records}\n")
        f.write("\n--- Type Breakdown ---\n")
        f.write(type_counts.to_string() + "\n")
        f.write("\n--- Severity Breakdown ---\n")
        f.write(severity_counts.to_string() + "\n")
        f.write("\n--- Spatial Breakdown ---\n")
        f.write(zone_counts.to_string() + "\n")
        if not ttc_df.empty:
            f.write("\n--- TTC Metrics ---\n")
            f.write(f"Average TTC: {ttc_df['value'].mean():.2f} s\n")
            f.write(f"Minimum TTC: {ttc_df['value'].min():.2f} s\n")
        if not pet_df.empty:
            f.write("\n--- PET Metrics ---\n")
            f.write(f"Average PET: {pet_df['value'].mean():.2f} s\n")
            f.write(f"Minimum PET: {pet_df['value'].min():.2f} s\n")
            
    print(f"\nSummary report saved to {summary_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze_pedestrian_conflicts()
