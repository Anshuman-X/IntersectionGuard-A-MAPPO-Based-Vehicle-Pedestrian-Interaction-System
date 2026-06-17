import os
import sys
import traci

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise Exception("SUMO_HOME not set")

sumo_cmd = [
    "sumo-gui",
    "-c",
    "simulation/t_intersection.sumocfg"
]

traci.start(sumo_cmd)

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    for vehicle in vehicles:

        speed = traci.vehicle.getSpeed(vehicle)

        position = traci.vehicle.getPosition(vehicle)

        print(
            f"Vehicle={vehicle} "
            f"Speed={speed:.2f} "
            f"Position={position}"
        )

traci.close()