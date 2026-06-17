import os
import sys
import traci
import math

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

sumo_cmd = [
    "sumo-gui",
    "-c",
    "simulation/t_intersection.sumocfg"
]

traci.start(sumo_cmd)

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    if len(vehicles) >= 2:

        v1 = vehicles[0]
        v2 = vehicles[1]

        pos1 = traci.vehicle.getPosition(v1)
        pos2 = traci.vehicle.getPosition(v2)

        speed1 = traci.vehicle.getSpeed(v1)

        distance = math.dist(pos1, pos2)

        if speed1 > 0:
            ttc = distance / speed1

            print(
                f"TTC between {v1} and {v2} = {ttc:.2f} sec"
            )

traci.close()