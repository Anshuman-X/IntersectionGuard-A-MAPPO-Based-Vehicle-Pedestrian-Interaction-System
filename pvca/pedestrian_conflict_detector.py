import os
import sys
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Ensure SUMO tools path is set up
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError("Please set the 'SUMO_HOME' environment variable to run the simulation.")

import traci

class PedestrianConflictDetector:
    """
    A modular class for analyzing vehicle-pedestrian conflicts at intersections.
    Designed for easy integration as a helper class in future MAPPO reinforcement learning environments.
    """
    def __init__(self, sumocfg_path, csv_output_path="results/pedestrian_conflicts.csv", 
                 junction_center=(100.0, 0.0), junction_radius=30.0, 
                 conflict_radius=20.0, ttc_threshold=3.0):
        
        self.sumocfg_path = sumocfg_path
        self.csv_output_path = csv_output_path
        self.junction_center = junction_center
        self.junction_radius = junction_radius
        self.conflict_radius = conflict_radius
        self.ttc_threshold = ttc_threshold
        
        # State metrics
        self.conflict_records = []
        
    def start_simulation(self, use_gui=False):
        """Initialize and start the TraCI SUMO simulation."""
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumocfg_path]
        traci.start(sumo_cmd)
        print(f"SUMO ({sumo_binary}) started successfully.")
        
        # Initialize the CSV file with the required headers
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
        with open(self.csv_output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "vehicle_id",
                "pedestrian_id",
                "distance",
                "vehicle_speed",
                "ttc"
            ])
            
    def is_near_crosswalks(self, position):
        """Check if a given 2D coordinate is within the junction/crosswalk radius."""
        return math.dist(position, self.junction_center) <= self.junction_radius
        
    def calculate_ttc(self, distance, speed):
        """
        Compute Time-to-Collision (TTC) in seconds.
        TTC = Distance / Vehicle Speed
        """
        if speed <= 0.05: # Threshold to avoid division by zero or extremely high TTC values
            return float('inf')
        return distance / speed

    def run_simulation_step(self):
        """Execute a single simulation step and detect vehicle-pedestrian conflicts."""
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        
        vehicles = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        
        # Filter vehicles and pedestrians near the crosswalks to reduce computational overhead
        near_vehicles = []
        vehicle_states = {}
        for v in vehicles:
            pos = traci.vehicle.getPosition(v)
            if self.is_near_crosswalks(pos):
                speed = traci.vehicle.getSpeed(v)
                near_vehicles.append(v)
                vehicle_states[v] = {"pos": pos, "speed": speed}
                
        near_pedestrians = []
        pedestrian_states = {}
        for p in pedestrians:
            pos = traci.person.getPosition(p)
            if self.is_near_crosswalks(pos):
                near_pedestrians.append(p)
                pedestrian_states[p] = {"pos": pos}
                
        # Analyze pairs of near-intersection actors
        step_conflicts = []
        for v in near_vehicles:
            v_pos = vehicle_states[v]["pos"]
            v_speed = vehicle_states[v]["speed"]
            
            for p in near_pedestrians:
                p_pos = pedestrian_states[p]["pos"]
                
                # Spatial distance check
                distance = math.dist(v_pos, p_pos)
                if distance <= self.conflict_radius:
                    # Calculate TTC
                    ttc = self.calculate_ttc(distance, v_speed)
                    
                    # Register conflict if below the safety threshold
                    if ttc <= self.ttc_threshold:
                        record = [
                            round(sim_time, 2),
                            v,
                            p,
                            round(distance, 2),
                            round(v_speed, 2),
                            round(ttc, 2)
                        ]
                        step_conflicts.append(record)
                        self.conflict_records.append(record)
                        
        # Append conflicts to CSV file in real time
        if step_conflicts:
            with open(self.csv_output_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(step_conflicts)
                
    def run_full_simulation(self, use_gui=False):
        """Run the simulation from start to finish."""
        self.start_simulation(use_gui)
        
        print("Running simulation steps...")
        while traci.simulation.getMinExpectedNumber() > 0:
            self.run_simulation_step()
            
        traci.close()
        print(f"Simulation finished. Raw conflict data saved to {self.csv_output_path}")

    def generate_statistics(self):
        """Analyze the conflict dataset and generate aggregate summary statistics."""
        if not os.path.exists(self.csv_output_path):
            print("Error: Conflict file does not exist. Please run the simulation first.")
            return None
            
        df = pd.read_csv(self.csv_output_path)
        
        if df.empty:
            print("No pedestrian conflicts were detected in the simulation.")
            return {
                "total_conflicts": 0,
                "avg_ttc": float('nan'),
                "min_ttc": float('nan'),
                "max_ttc": float('nan')
            }
            
        stats = {
            "total_conflicts": len(df),
            "avg_ttc": round(df["ttc"].mean(), 2),
            "min_ttc": round(df["ttc"].min(), 2),
            "max_ttc": round(df["ttc"].max(), 2)
        }
        
        print("\n" + "="*40)
        print("   VEHICLE-PEDESTRIAN TTC STATISTICS")
        print("="*40)
        print(f"Total Pedestrian Conflicts : {stats['total_conflicts']}")
        print(f"Average TTC (seconds)      : {stats['avg_ttc']}s")
        print(f"Minimum TTC (seconds)      : {stats['min_ttc']}s")
        print(f"Maximum TTC (seconds)      : {stats['max_ttc']}s")
        print("="*40 + "\n")
        
        return stats

    def generate_ttc_histogram(self, output_image_path="results/pedestrian_ttc_histogram.png"):
        """Plot and save a histogram of the TTC conflict values."""
        if not os.path.exists(self.csv_output_path):
            print("Error: Conflict file does not exist. Cannot plot histogram.")
            return
            
        df = pd.read_csv(self.csv_output_path)
        if df.empty:
            print("No data to plot.")
            return
            
        plt.figure(figsize=(8, 5))
        plt.hist(df["ttc"], bins=20, range=(0, self.ttc_threshold), color="#d9534f", edgecolor="black", alpha=0.85, rwidth=0.9)
        plt.title("Distribution of Vehicle-Pedestrian Time-to-Collision (TTC)")
        plt.xlabel("TTC (seconds)")
        plt.ylabel("Frequency (Steps)")
        plt.xlim(0, self.ttc_threshold)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        print(f"TTC histogram plot successfully saved to {output_image_path}")

if __name__ == "__main__":
    # Path configuration
    sumocfg = "simulation/t_intersection.sumocfg"
    
    # Instantiate detector
    detector = PedestrianConflictDetector(
        sumocfg_path=sumocfg,
        csv_output_path="results/pedestrian_conflicts.csv",
        junction_center=(100.0, 0.0), # Central T-intersection coordinates
        junction_radius=30.0,         # 30m detection zone around crosswalks
        conflict_radius=20.0,         # Calculate TTC when actors are within 20m of each other
        ttc_threshold=3.0             # Limit logging to safety-critical events where TTC <= 3.0s
    )
    
    # Run the detection pipeline
    detector.run_full_simulation(use_gui=False)
    detector.generate_statistics()
    detector.generate_ttc_histogram()
