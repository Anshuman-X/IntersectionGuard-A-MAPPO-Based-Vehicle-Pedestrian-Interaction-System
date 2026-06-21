import os
import sys
import traci

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise Exception("SUMO_HOME not set")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Live vehicle tracker")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()

    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c",
        "simulation/t_intersection.sumocfg"
    ]

    traci.start(sumo_cmd)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()

        for vehicle in vehicles:
            try:
                speed = traci.vehicle.getSpeed(vehicle)
                position = traci.vehicle.getPosition(vehicle)
                print(
                    f"Vehicle={vehicle} "
                    f"Speed={speed:.2f} "
                    f"Position={position}"
                )
            except traci.exceptions.TraCIException:
                continue

    traci.close()

if __name__ == "__main__":
    main()