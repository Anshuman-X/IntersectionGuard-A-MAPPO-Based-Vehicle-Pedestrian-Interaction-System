import os
import random
import pandas as pd

def generate_pedestrian_demand(file_path, total_time=1000, seed=42):
    random.seed(seed)
    
    # 1. Read calibration data from workspace data folder or D drive fallback
    calib_paths = [
        "data/sumo_pedestrian_calibration.csv",
        r"D:\RESEARCH INTERNSHIP NITT\DATA SETS\cleaned_data\TRAINING_DATA\sumo_pedestrian_calibration.csv"
    ]
    
    crossing_rate = 80.98
    avg_gap_acceptance = 3.62
    avg_waiting_time = 0.69
    
    for path in calib_paths:
        if os.path.exists(path):
            try:
                calib_df = pd.read_csv(path)
                crossing_rate = float(calib_df.loc[0, "crossing_rate"])
                avg_gap_acceptance = float(calib_df.loc[0, "avg_gap_acceptance"])
                avg_waiting_time = float(calib_df.loc[0, "avg_waiting_time"])
                print(f"Loaded pedestrian calibration from {path}: crossing_rate={crossing_rate:.2f}, avg_gap_acceptance={avg_gap_acceptance:.2f}")
                break
            except Exception as e:
                print(f"Error reading {path}: {e}")
                
    # Calculate calibrated number of pedestrians
    num_pedestrians = int(crossing_rate * (total_time / 60.0))
    
    # Define pedestrian OD paths that require crossing the junction
    paths = [
        # Crosses West leg
        ("west_to_west", ["west_center", "center_west"]),
        # Crosses East leg
        ("east_to_east", ["east_center", "center_east"]),
        # Crosses North leg
        ("north_to_north", ["north_center", "center_north"]),
        # Crosses West leg and North leg
        ("west_to_north", ["west_center", "center_north"]),
        # Crosses East leg and North leg
        ("east_to_north", ["east_center", "center_north"]),
        # Crosses North leg and West leg
        ("north_to_west", ["north_center", "center_west"]),
        # Crosses North leg and East leg
        ("north_to_east", ["north_center", "center_east"]),
    ]
    
    # Generate departure times
    depart_times = sorted([random.uniform(5, total_time - 50) for _ in range(num_pedestrians)])
    
    with open(file_path, "w") as f:
        f.write("<routes>\n\n")
        f.write("    <!-- Pedestrian Type Definition -->\n")
        # Calibrate crossing decision gap acceptance via jmCrossingGap
        f.write(f'    <vType id="pedestrian_type" vClass="pedestrian" speedDev="0.2" length="0.5" width="0.5" minGap="0.2" jmCrossingGap="{avg_gap_acceptance:.2f}"/>\n\n')
        
        for i, depart in enumerate(depart_times):
            path_name, edges = random.choice(paths)
            edges_str = " ".join(edges)
            
            f.write(f'    <person id="ped_{i}" type="pedestrian_type" depart="{depart:.2f}">\n')
            f.write(f'        <walk edges="{edges_str}"/>\n')
            f.write('    </person>\n\n')
            
        f.write("</routes>\n")
        
    print(f"Successfully generated {num_pedestrians} pedestrians in {file_path}")

if __name__ == "__main__":
    generate_pedestrian_demand("demand/pedestrians.rou.xml")
