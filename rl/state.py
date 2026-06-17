import math
import traci

class StateExtractor:
    """
    Extracts and maps SUMO simulation data to local observation spaces for each 
    cooperative agent (West, East, and North approach controllers) at the T-intersection.
    """
    def __init__(self, junction_center=(100.0, 0.0), detection_radius=35.0):
        self.junction_center = junction_center
        self.detection_radius = detection_radius
        
        # Define approach edges corresponding to each agent
        self.approach_edges = {
            "west_controller": ["west_center"],
            "east_controller": ["east_center"],
            "north_controller": ["north_center"]
        }
        
        # Define crossing coordinates bounding boxes
        self.crossing_zones = {
            "west_controller": {"x_min": 92.8, "x_max": 96.8, "y_min": -3.5, "y_max": 3.5},
            "east_controller": {"x_min": 103.2, "x_max": 107.2, "y_min": -3.5, "y_max": 3.5},
            "north_controller": {"x_min": 96.5, "x_max": 103.5, "y_min": 3.2, "y_max": 7.2}
        }

    def get_agent_vehicles(self, agent_id, vehicles):
        """Get vehicles that belong to a specific approach or agent control zone."""
        agent_vehs = []
        for v in vehicles:
            edge = traci.vehicle.getRoadID(v)
            # Check if vehicle is on the agent's approach edge or within the junction zone
            if edge in self.approach_edges[agent_id]:
                agent_vehs.append(v)
            else:
                # Fallback spatial check for vehicles inside junction heading to respective directions
                pos = traci.vehicle.getPosition(v)
                if math.dist(pos, self.junction_center) <= self.detection_radius:
                    angle = traci.vehicle.getAngle(v) # Degrees, 0 is North, 90 East, 180 South, 270 West
                    if agent_id == "west_controller" and (225 <= angle <= 315):
                        agent_vehs.append(v)
                    elif agent_id == "east_controller" and (45 <= angle <= 135):
                        agent_vehs.append(v)
                    elif agent_id == "north_controller" and (315 < angle or angle < 45 or 135 < angle < 225):
                        agent_vehs.append(v)
        return list(set(agent_vehs))

    def get_agent_pedestrians(self, agent_id, pedestrians):
        """Get pedestrians currently occupying the crossing zone controlled by this agent."""
        agent_peds = []
        bbox = self.crossing_zones[agent_id]
        for p in pedestrians:
            pos = traci.person.getPosition(p)
            # Check if pedestrian is inside the crossing bounding box
            if bbox["x_min"] <= pos[0] <= bbox["x_max"] and bbox["y_min"] <= pos[1] <= bbox["y_max"]:
                agent_peds.append(p)
        return agent_peds

    def get_agent_observation(self, agent_id, vehicles, pedestrians, vehicle_ttc_threshold=3.0):
        """
        Extract the 5-feature observation vector for a given agent:
        [vehicle_count, pedestrian_count, average_ttc, average_speed, conflict_count]
        """
        agent_vehs = self.get_agent_vehicles(agent_id, vehicles)
        agent_peds = self.get_agent_pedestrians(agent_id, pedestrians)
        
        veh_count = len(agent_vehs)
        ped_count = len(agent_peds)
        
        # Compute average speed
        avg_speed = 0.0
        if veh_count > 0:
            avg_speed = sum(traci.vehicle.getSpeed(v) for v in agent_vehs) / veh_count
            
        # Compute conflicts and TTC
        conflict_count = 0
        ttc_values = []
        
        # 1. Vehicle-Pedestrian TTC Conflicts
        for v in agent_vehs:
            v_pos = traci.vehicle.getPosition(v)
            v_speed = traci.vehicle.getSpeed(v)
            for p in agent_peds:
                p_pos = traci.person.getPosition(p)
                dist = math.dist(v_pos, p_pos)
                
                # Proximity criteria
                if dist <= 20.0 and v_speed > 0.1:
                    ttc = dist / v_speed
                    if ttc <= vehicle_ttc_threshold:
                        ttc_values.append(ttc)
                        conflict_count += 1
                        
        # 2. Vehicle-Vehicle TTC Conflicts (among vehicles controlled by this agent)
        for i in range(len(agent_vehs)):
            v1 = agent_vehs[i]
            pos1 = traci.vehicle.getPosition(v1)
            speed1 = traci.vehicle.getSpeed(v1)
            for j in range(i + 1, len(agent_vehs)):
                v2 = agent_vehs[j]
                pos2 = traci.vehicle.getPosition(v2)
                dist = math.dist(pos1, pos2)
                
                # Simple car-following headway or projected TTC
                rel_speed = abs(speed1 - traci.vehicle.getSpeed(v2))
                if dist <= 15.0 and rel_speed > 0.1:
                    ttc = dist / rel_speed
                    if ttc <= vehicle_ttc_threshold:
                        ttc_values.append(ttc)
                        conflict_count += 1
                        
        avg_ttc = sum(ttc_values) / len(ttc_values) if ttc_values else vehicle_ttc_threshold
        
        return [
            float(veh_count),
            float(ped_count),
            float(round(avg_ttc, 2)),
            float(round(avg_speed, 2)),
            float(conflict_count)
        ]
