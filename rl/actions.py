import traci

class ActionExecutor:
    """
    Translates discrete actions chosen by the reinforcement learning agents 
    into low-level SUMO vehicle dynamics controls via TraCI.
    """
    def __init__(self, slow_speed=4.0, stop_speed=0.0):
        self.slow_speed = slow_speed
        self.stop_speed = stop_speed
        
        # Track previous stop states to avoid resetting unchanged stops
        self.active_stops = {}

    def execute_action(self, agent_id, action, agent_vehicles, agent_pedestrians):
        """
        Execute one of 4 discrete actions on the vehicles controlled by this agent:
        - 0 (Do Nothing): Vehicles flow at default speed and behavior.
        - 1 (Slow Vehicles): Set max speed to slow_speed (4.0 m/s / ~14.4 km/h).
        - 2 (Stop Vehicles): Forces vehicles to stop immediately.
        - 3 (Prioritize Pedestrians): Yield to pedestrians (stop if pedestrian is crossing, else proceed).
        """
        for v in agent_vehicles:
            try:
                # 1. Do Nothing
                if action == 0:
                    # Restore default max speed of the vehicle's type
                    v_type = traci.vehicle.getTypeID(v)
                    default_max_speed = traci.vehicletype.getMaxSpeed(v_type)
                    traci.vehicle.setMaxSpeed(v, default_max_speed)
                    # Restore default model control
                    traci.vehicle.setSpeed(v, -1)
                
                # 2. Slow Vehicles
                elif action == 1:
                    traci.vehicle.setMaxSpeed(v, self.slow_speed)
                    traci.vehicle.setSpeed(v, -1)
                
                # 3. Stop Vehicles
                elif action == 2:
                    # Force decelerate to 0
                    traci.vehicle.setSpeed(v, self.stop_speed)
                
                # 4. Prioritize Pedestrians (Dynamic Yielding)
                elif action == 3:
                    if len(agent_pedestrians) > 0:
                        # Pedestrians are in the crossing; force stop
                        traci.vehicle.setSpeed(v, self.stop_speed)
                    else:
                        # Crosswalk is empty; vehicles can flow at default speed
                        v_type = traci.vehicle.getTypeID(v)
                        default_max_speed = traci.vehicletype.getMaxSpeed(v_type)
                        traci.vehicle.setMaxSpeed(v, default_max_speed)
                        traci.vehicle.setSpeed(v, -1)
                        
            except traci.exceptions.TraCIException:
                # Vehicle might have exited the simulation or changed state in this step
                continue
