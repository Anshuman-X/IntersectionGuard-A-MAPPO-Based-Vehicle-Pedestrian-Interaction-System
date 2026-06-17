import pandas as pd
import os

def calculate_pet_statistics(csv_path="results/pet_conflicts.csv"):
    """Load pet_conflicts.csv and print the aggregate safety statistics."""
    if not os.path.exists(csv_path):
        print(f"Error: Conflict file {csv_path} not found. Please run the simulation first.")
        return None
        
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("No PET conflicts found in the dataset.")
        return {
            "total": 0,
            "avg": float('nan'),
            "min": float('nan'),
            "max": float('nan')
        }
        
    stats = {
        "total": len(df),
        "avg": round(df["PET"].mean(), 2),
        "min": round(df["PET"].min(), 2),
        "max": round(df["PET"].max(), 2)
    }
    
    print("\n" + "="*40)
    print("      POST ENCROACHMENT TIME (PET) STATS")
    print("="*40)
    print(f"Total PET Conflicts : {stats['total']}")
    print(f"Average PET         : {stats['avg']:.2f} seconds")
    print(f"Minimum PET         : {stats['min']:.2f} seconds")
    print(f"Maximum PET         : {stats['max']:.2f} seconds")
    print("="*40 + "\n")
    
    return stats

if __name__ == "__main__":
    calculate_pet_statistics()
