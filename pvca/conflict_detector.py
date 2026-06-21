import os
import sys
import traci
import math
import csv

# SUMO Tools Path
if "SUMO_HOME" in os.environ:
    sys.path.append(
        os.path.join(os.environ["SUMO_HOME"], "tools")
    )
else:
    raise EnvironmentError("SUMO_HOME not found")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Vehicle-vehicle conflict detector")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()

    # SUMO Configuration
    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c",
        "simulation/t_intersection.sumocfg"
    ]

    # Create CSV and write header once
    with open("results/conflicts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "vehicle_1",
            "vehicle_2",
            "distance",
            "speed",
            "ttc"
        ])

    # Start SUMO
    traci.start(sumo_cmd)

    CONFLICT_THRESHOLD = 3.0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1 = vehicles[i]
                v2 = vehicles[j]

                try:
                    pos1 = traci.vehicle.getPosition(v1)
                    pos2 = traci.vehicle.getPosition(v2)
                    speed1 = traci.vehicle.getSpeed(v1)
                    distance = math.dist(pos1, pos2)

                    if speed1 > 0:
                        ttc = distance / speed1
                        if ttc < CONFLICT_THRESHOLD:
                            sim_time = traci.simulation.getTime()
                            print(f"CONFLICT: {v1} <-> {v2} | TTC={ttc:.2f}")

                            # Append conflict to CSV
                            with open("results/conflicts.csv", "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    sim_time,
                                    v1,
                                    v2,
                                    round(distance, 2),
                                    round(speed1, 2),
                                    round(ttc, 2)
                                ])
                except traci.exceptions.TraCIException:
                    continue

    # Close SUMO
    traci.close()
    print("\nConflict data saved to results/conflicts.csv")

if __name__ == "__main__":
    main()