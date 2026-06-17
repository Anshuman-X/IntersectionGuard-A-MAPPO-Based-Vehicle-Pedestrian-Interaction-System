import os
import sys
import csv
import math

# Ensure SUMO tools path is set up
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError("Please set the 'SUMO_HOME' environment variable to run the simulation.")

import traci

# Conflict zones at pedestrian crossings
# Junction center is at (100.0, 0.0) in compiled network coordinates.
CONFLICT_ZONES = {
    "West": {
        "x_min": 92.8, "x_max": 96.8,
        "y_min": -3.5, "y_max": 3.5
    },
    "East": {
        "x_min": 103.2, "x_max": 107.2,
        "y_min": -3.5, "y_max": 3.5
    },
    "North": {
        "x_min": 96.5, "x_max": 103.5,
        "y_min": 3.2,  "y_max": 7.2
    }
}

PET_THRESHOLD = 5.0  # Max PET in seconds to register a conflict

def get_occupied_zone(x, y):
    """Return the name of the conflict zone if coordinates are inside, else None."""
    for zone_name, bbox in CONFLICT_ZONES.items():
        if bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]:
            return zone_name
    return None

def run_pet_analysis():
    sumocfg = "simulation/t_intersection.sumocfg"
    sumo_cmd = ["sumo", "-c", sumocfg]
    traci.start(sumo_cmd)
    print("SUMO started for PET Analysis...")
    
    # Initialize output file
    os.makedirs("results", exist_ok=True)
    csv_file = "results/pet_conflicts.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "vehicle_id",
            "pedestrian_id",
            "vehicle_exit_time",
            "pedestrian_entry_time",
            "PET"
        ])
        
    # State tracking dictionaries
    # occupied_by_veh[zone] = set(vehicle_ids)
    occupied_by_veh = {zone: set() for zone in CONFLICT_ZONES}
    # occupied_by_ped[zone] = set(pedestrian_ids)
    occupied_by_ped = {zone: set() for zone in CONFLICT_ZONES}
    
    # last_vehicle_exit[zone] = (vehicle_id, exit_time)
    last_vehicle_exit = {zone: None for zone in CONFLICT_ZONES}
    
    pet_count = 0
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        
        vehicles = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        
        # Determine current occupancy in this step
        current_veh_occupancy = {zone: set() for zone in CONFLICT_ZONES}
        for v in vehicles:
            pos = traci.vehicle.getPosition(v)
            zone = get_occupied_zone(pos[0], pos[1])
            if zone:
                current_veh_occupancy[zone].add(v)
                
        current_ped_occupancy = {zone: set() for zone in CONFLICT_ZONES}
        for p in pedestrians:
            pos = traci.person.getPosition(p)
            zone = get_occupied_zone(pos[0], pos[1])
            if zone:
                current_ped_occupancy[zone].add(p)
                
        # Process each conflict zone
        for zone in CONFLICT_ZONES:
            # 1. Detect when a vehicle exits the conflict zone
            for v in list(occupied_by_veh[zone]):
                if v not in current_veh_occupancy[zone]:
                    # Vehicle exited the zone in this step
                    exit_time = sim_time
                    last_vehicle_exit[zone] = (v, exit_time)
                    
            # 2. Detect when a pedestrian enters the same conflict zone
            for p in current_ped_occupancy[zone]:
                if p not in occupied_by_ped[zone]:
                    # Pedestrian entered the zone in this step
                    entry_time = sim_time
                    
                    # Check if there was a preceding vehicle exit
                    if last_vehicle_exit[zone] is not None:
                        v_id, exit_time = last_vehicle_exit[zone]
                        pet = entry_time - exit_time
                        
                        # Register conflict if PET is positive and below threshold
                        if 0.0 < pet <= PET_THRESHOLD:
                            # Log PET Conflict
                            with open(csv_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    round(sim_time, 2),
                                    v_id,
                                    p,
                                    round(exit_time, 2),
                                    round(entry_time, 2),
                                    round(pet, 2)
                                ])
                            pet_count += 1
                            print(f"[{sim_time:.1f}s] PET Conflict in {zone} zone: Veh {v_id} -> Ped {p} | PET = {pet:.2f}s")
                            
                            # Clear vehicle exit to prevent re-matching with other entering pedestrians in this step
                            last_vehicle_exit[zone] = None
                            
            # Update history sets for the next step
            occupied_by_veh[zone] = current_veh_occupancy[zone].copy()
            occupied_by_ped[zone] = current_ped_occupancy[zone].copy()
            
    traci.close()
    print(f"\nSimulation completed. Total PET conflicts logged: {pet_count}")

if __name__ == "__main__":
    run_pet_analysis()
