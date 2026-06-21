import math
import traci

class StateExtractor:
    """
    Extracts and maps SUMO simulation data to local observation spaces for each 
    cooperative agent (West, East, and North approach controllers) at the T-intersection.
    Optimized with state caching to eliminate TraCI communication bottlenecks.
    """
    def __init__(self, junction_center=(100.0, 0.0), detection_radius=35.0):
        self.junction_center = junction_center
        self.detection_radius = detection_radius
        
        # State cache to store TraCI queries once per simulation step
        self.cache = {}
        
        # Define approach edges corresponding to each agent
        self.approach_edges = {
            "west_controller": ["west_center"],
            "east_controller": ["east_center"],
            "north_controller": ["north_center"]
        }
        
        # Define crossing coordinates bounding boxes
        # Proactive yielding (Task 6): Extended boundaries outwards by 2.0 meters
        # to capture pedestrians waiting on the sidewalks before they enter the road.
        self.crossing_zones = {
            "west_controller":  {"x_min": 92.8,  "x_max": 96.8,  "y_min": -5.5, "y_max": 5.5},
            "east_controller":  {"x_min": 103.2, "x_max": 107.2, "y_min": -5.5, "y_max": 5.5},
            "north_controller": {"x_min": 94.5,  "x_max": 105.5, "y_min": 3.2,  "y_max": 7.2}
        }
        
        # Calibrated gap acceptance baseline (Task 1)
        self.gap_acceptance_val = 3.62

    def update_cache(self, vehicles, pedestrians):
        """Query all TraCI attributes in a single pass to resolve communication bottlenecks."""
        self.cache = {
            "veh_road": {},
            "veh_pos": {},
            "veh_speed": {},
            "veh_angle": {},
            "veh_waiting": {},
            "ped_pos": {},
            "ped_waiting": {}
        }
        for v in vehicles:
            try:
                self.cache["veh_road"][v] = traci.vehicle.getRoadID(v)
                self.cache["veh_pos"][v] = traci.vehicle.getPosition(v)
                self.cache["veh_speed"][v] = traci.vehicle.getSpeed(v)
                self.cache["veh_angle"][v] = traci.vehicle.getAngle(v)
                self.cache["veh_waiting"][v] = traci.vehicle.getWaitingTime(v)
            except traci.exceptions.TraCIException:
                continue
                
        for p in pedestrians:
            try:
                self.cache["ped_pos"][p] = traci.person.getPosition(p)
                self.cache["ped_waiting"][p] = traci.person.getWaitingTime(p)
            except traci.exceptions.TraCIException:
                continue

    def clear_cache(self):
        """Clear cache at step end."""
        self.cache = {}

    def get_agent_vehicles(self, agent_id, vehicles):
        """Get vehicles that belong to a specific approach or agent control zone from cache."""
        agent_vehs = []
        for v in vehicles:
            if v not in self.cache["veh_road"]:
                continue
            edge = self.cache["veh_road"][v]
            if edge in self.approach_edges[agent_id]:
                agent_vehs.append(v)
            else:
                # Fallback spatial check for vehicles inside junction heading to respective directions
                pos = self.cache["veh_pos"][v]
                if math.dist(pos, self.junction_center) <= self.detection_radius:
                    angle = self.cache["veh_angle"][v] # Degrees, 0 is North, 90 East, 180 South, 270 West
                    if agent_id == "west_controller" and (225 <= angle <= 315):
                        agent_vehs.append(v)
                    elif agent_id == "east_controller" and (45 <= angle <= 135):
                        agent_vehs.append(v)
                    elif agent_id == "north_controller" and (315 < angle or angle < 45 or 135 < angle < 225):
                        agent_vehs.append(v)
        return list(set(agent_vehs))

    def get_agent_pedestrians(self, agent_id, pedestrians):
        """Get pedestrians occupying crossing or waiting to cross from cache (proactive yielding)."""
        agent_peds = []
        bbox = self.crossing_zones[agent_id]
        for p in pedestrians:
            if p not in self.cache["ped_pos"]:
                continue
            pos = self.cache["ped_pos"][p]
            # Check if pedestrian is inside the extended crossing bounding box
            if bbox["x_min"] <= pos[0] <= bbox["x_max"] and bbox["y_min"] <= pos[1] <= bbox["y_max"]:
                agent_peds.append(p)
        return agent_peds

    def get_agent_observation(self, agent_id, vehicles, pedestrians, step_pet=5.0, vehicle_ttc_threshold=3.0):
        """
        Extract the 9-feature observation vector for a given agent (Task 2):
        [avg_speed, avg_pos, avg_direction, density, avg_waiting_time, step_pet, avg_ttc, gap_acceptance, crossing_status]
        """
        # Auto-initialize cache if not pre-populated
        if not self.cache:
            self.update_cache(vehicles, pedestrians)
            
        agent_vehs = self.get_agent_vehicles(agent_id, vehicles)
        agent_peds = self.get_agent_pedestrians(agent_id, pedestrians)
        
        veh_count = len(agent_vehs)
        ped_count = len(agent_peds)
        
        # 1. Vehicle Features
        avg_speed = 0.0
        avg_pos = self.detection_radius
        avg_direction = 0.0
        
        if veh_count > 0:
            speeds = [self.cache["veh_speed"][v] for v in agent_vehs if v in self.cache["veh_speed"]]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            
            distances = [math.dist(self.cache["veh_pos"][v], self.junction_center) for v in agent_vehs if v in self.cache["veh_pos"]]
            avg_pos = sum(distances) / len(distances) if distances else self.detection_radius
            
            angles = [self.cache["veh_angle"][v] for v in agent_vehs if v in self.cache["veh_angle"]]
            avg_direction = sum(angles) / len(angles) if angles else 0.0
            
        density = float(veh_count) / self.detection_radius
        
        # 2. Pedestrian Features
        avg_waiting_time = 0.0
        if ped_count > 0:
            waiting_times = [self.cache["ped_waiting"][p] for p in agent_peds if p in self.cache["ped_waiting"]]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0
            
        # Compute vehicle-pedestrian TTC in this agent's zone
        ttc_values = []
        for v in agent_vehs:
            if v not in self.cache["veh_pos"] or v not in self.cache["veh_speed"]:
                continue
            v_pos = self.cache["veh_pos"][v]
            v_speed = self.cache["veh_speed"][v]
            for p in agent_peds:
                if p not in self.cache["ped_pos"]:
                    continue
                p_pos = self.cache["ped_pos"][p]
                dist = math.dist(v_pos, p_pos)
                
                # Proximity criteria
                if dist <= 20.0 and v_speed > 0.1:
                    ttc = dist / v_speed
                    if ttc <= vehicle_ttc_threshold:
                        ttc_values.append(ttc)
                        
        avg_ttc = sum(ttc_values) / len(ttc_values) if ttc_values else vehicle_ttc_threshold
        crossing_status = 1.0 if ped_count > 0 else 0.0
        
        return [
            float(round(avg_speed, 2)),
            float(round(avg_pos, 2)),
            float(round(avg_direction, 2)),
            float(round(density, 3)),
            float(round(avg_waiting_time, 2)),
            float(round(step_pet, 2)),
            float(round(avg_ttc, 2)),
            float(self.gap_acceptance_val),
            float(crossing_status)
        ]
