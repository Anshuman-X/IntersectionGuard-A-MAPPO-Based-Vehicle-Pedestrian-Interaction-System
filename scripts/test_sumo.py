import os
import sys

print("Checking SUMO...")

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

    print("SUMO_HOME found:")
    print(os.environ["SUMO_HOME"])

else:
    print("SUMO_HOME NOT FOUND")
    exit()

try:
    import traci
    print("TraCI imported successfully")
except Exception as e:
    print("TraCI Error:", e)