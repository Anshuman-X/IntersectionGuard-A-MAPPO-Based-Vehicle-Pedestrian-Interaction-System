import os
import sys
import math
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
    Detects TTC, Overlap, and PET conflicts, categorizing them by zone and severity.
    """
    def __init__(self, sumocfg_path, csv_output_path="results/pedestrian_conflicts.csv", 
                 junction_center=(100.0, 0.0), junction_radius=30.0, 
                 conflict_radius=20.0, ttc_threshold=3.0, pet_threshold=5.0):
        
        self.sumocfg_path = sumocfg_path
        self.csv_output_path = csv_output_path
        self.junction_center = junction_center
        self.junction_radius = junction_radius
        self.conflict_radius = conflict_radius
        self.ttc_threshold = ttc_threshold
        self.pet_threshold = pet_threshold
        
        # Conflict zone configurations matching state.py and pet.py
        self.conflict_zones = {
            "West":  {"x_min": 92.8,  "x_max": 96.8,  "y_min": -3.5, "y_max": 3.5},
            "East":  {"x_min": 103.2, "x_max": 107.2, "y_min": -3.5, "y_max": 3.5},
            "North": {"x_min": 96.5,  "x_max": 103.5, "y_min":  3.2, "y_max": 7.2},
        }
        
        # Tracking dictionaries for PET
        self.occupied_by_veh = {zone: set() for zone in self.conflict_zones}
        self.occupied_by_ped = {zone: set() for zone in self.conflict_zones}
        self.last_vehicle_exit = {zone: None for zone in self.conflict_zones}
        
        self.conflict_records = []
        
    def get_occupied_zone(self, x, y):
        """Return the name of the conflict zone if coordinates are inside, else 'General (None)'."""
        for zone_name, bbox in self.conflict_zones.items():
            if bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]:
                return zone_name
        return "General (None)"
        
    def start_simulation(self, use_gui=False):
        """Initialize and start the TraCI SUMO simulation."""
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumocfg_path]
        traci.start(sumo_cmd)
        print(f"SUMO ({sumo_binary}) started successfully.")
        
        # Initialize the CSV file with the required headers for the analyzer/visualizer
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
        with open(self.csv_output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "conflict_type",
                "severity",
                "zone",
                "value",
                "distance",
                "vehicle_speed",
                "pedestrian_speed",
                "vehicle_id",
                "pedestrian_id"
            ])
            
    def is_near_crosswalks(self, position):
        """Check if a given 2D coordinate is within the junction/crosswalk radius."""
        return math.dist(position, self.junction_center) <= self.junction_radius
        
    def calculate_ttc(self, distance, speed):
        """Compute Time-to-Collision (TTC) in seconds."""
        if speed <= 0.05:
            return float('inf')
        return distance / speed

    def run_simulation_step(self):
        """Execute a single simulation step and detect vehicle-pedestrian conflicts (TTC, Overlap, PET)."""
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        
        vehicles = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        
        # 1. Track current zone occupancies for PET
        current_veh_occupancy = {zone: set() for zone in self.conflict_zones}
        current_ped_occupancy = {zone: set() for zone in self.conflict_zones}
        
        near_vehicles = []
        vehicle_states = {}
        for v in vehicles:
            try:
                pos = traci.vehicle.getPosition(v)
                zone = self.get_occupied_zone(pos[0], pos[1])
                if zone != "General (None)":
                    current_veh_occupancy[zone].add(v)
                
                if self.is_near_crosswalks(pos):
                    speed = traci.vehicle.getSpeed(v)
                    near_vehicles.append(v)
                    vehicle_states[v] = {"pos": pos, "speed": speed}
            except traci.exceptions.TraCIException:
                continue
                
        near_pedestrians = []
        pedestrian_states = {}
        for p in pedestrians:
            try:
                pos = traci.person.getPosition(p)
                zone = self.get_occupied_zone(pos[0], pos[1])
                if zone != "General (None)":
                    current_ped_occupancy[zone].add(p)
                
                if self.is_near_crosswalks(pos):
                    speed = traci.person.getSpeed(p)
                    near_pedestrians.append(p)
                    pedestrian_states[p] = {"pos": pos, "speed": speed}
            except traci.exceptions.TraCIException:
                continue
                
        step_conflicts = []
        
        # 2. Detect TTC and Overlap conflicts
        for v in near_vehicles:
            v_pos = vehicle_states[v]["pos"]
            v_speed = vehicle_states[v]["speed"]
            
            for p in near_pedestrians:
                p_pos = pedestrian_states[p]["pos"]
                p_speed = pedestrian_states[p]["speed"]
                
                distance = math.dist(v_pos, p_pos)
                if distance <= self.conflict_radius:
                    zone = self.get_occupied_zone(p_pos[0], p_pos[1])
                    
                    # Check for Overlap conflict
                    if distance < 1.5:
                        record = [
                            round(sim_time, 2),
                            "Overlap",
                            "Critical",
                            zone,
                            0.0,
                            round(distance, 2),
                            round(v_speed, 2),
                            round(p_speed, 2),
                            v,
                            p
                        ]
                        step_conflicts.append(record)
                        self.conflict_records.append(record)
                    else:
                        # Check for TTC conflict
                        ttc = self.calculate_ttc(distance, v_speed)
                        if ttc <= self.ttc_threshold:
                            severity = "Low"
                            if ttc < 1.0:
                                severity = "Critical"
                            elif ttc < 2.0:
                                severity = "Moderate"
                                
                            record = [
                                round(sim_time, 2),
                                "TTC",
                                severity,
                                zone,
                                round(ttc, 2),
                                round(distance, 2),
                                round(v_speed, 2),
                                round(p_speed, 2),
                                v,
                                p
                            ]
                            step_conflicts.append(record)
                            self.conflict_records.append(record)
                            
        # 3. Detect PET conflicts
        for zone in self.conflict_zones:
            # Detect when a vehicle exits the zone
            for v in list(self.occupied_by_veh[zone]):
                if v not in current_veh_occupancy[zone]:
                    # Vehicle exited in this step
                    self.last_vehicle_exit[zone] = (v, sim_time)
                    
            # Detect when a pedestrian enters the zone
            for p in current_ped_occupancy[zone]:
                if p not in self.occupied_by_ped[zone]:
                    # Pedestrian entered in this step
                    if self.last_vehicle_exit[zone] is not None:
                        v_id, exit_time = self.last_vehicle_exit[zone]
                        pet = sim_time - exit_time
                        
                        if 0.0 < pet <= self.pet_threshold:
                            # Log PET conflict
                            severity = "Low"
                            if pet < 1.5:
                                severity = "Critical"
                            elif pet < 3.0:
                                severity = "Moderate"
                                
                            try:
                                v_speed = traci.vehicle.getSpeed(v_id) if v_id in vehicles else 0.0
                                p_speed = traci.person.getSpeed(p) if p in pedestrians else 0.0
                            except traci.exceptions.TraCIException:
                                v_speed = 0.0
                                p_speed = 0.0
                                
                            record = [
                                round(sim_time, 2),
                                "PET",
                                severity,
                                zone,
                                round(pet, 2),
                                0.0, # distance not directly defined at entry event
                                round(v_speed, 2),
                                round(p_speed, 2),
                                v_id,
                                p
                            ]
                            step_conflicts.append(record)
                            self.conflict_records.append(record)
                            
                            # Consume the exit record
                            self.last_vehicle_exit[zone] = None
                            
            # Update histories for next step
            self.occupied_by_veh[zone] = current_veh_occupancy[zone].copy()
            self.occupied_by_ped[zone] = current_ped_occupancy[zone].copy()
                        
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
        print(f"Simulation finished. Calibrated conflict data saved to {self.csv_output_path}")

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
            
        ttc_df = df[df["conflict_type"] == "TTC"]
        avg_ttc = round(ttc_df["value"].mean(), 2) if not ttc_df.empty else float('nan')
        min_ttc = round(ttc_df["value"].min(), 2) if not ttc_df.empty else float('nan')
        max_ttc = round(ttc_df["value"].max(), 2) if not ttc_df.empty else float('nan')
        
        stats = {
            "total_conflicts": len(df),
            "avg_ttc": avg_ttc,
            "min_ttc": min_ttc,
            "max_ttc": max_ttc
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
            
        ttc_df = df[df["conflict_type"] == "TTC"]
        if ttc_df.empty:
            print("No TTC conflict records to plot.")
            return
            
        plt.figure(figsize=(8, 5))
        plt.hist(ttc_df["value"], bins=20, range=(0, self.ttc_threshold), color="#d9534f", edgecolor="black", alpha=0.85, rwidth=0.9)
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
