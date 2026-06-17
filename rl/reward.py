class RewardCalculator:
    """
    Computes step-based rewards for cooperative reinforcement learning agents.
    Balances safety (minimizing TTC/PET conflicts) and efficiency (maximizing throughput).
    """
    def __init__(self, w_speed=0.2, w_veh_conflict=2.0, w_ped_conflict=10.0, w_ttc_penalty=1.0, ttc_safe_limit=3.0):
        self.w_speed = w_speed
        self.w_veh_conflict = w_veh_conflict
        self.w_ped_conflict = w_ped_conflict
        self.w_ttc_penalty = w_ttc_penalty
        self.ttc_safe_limit = ttc_safe_limit

    def calculate_reward(self, agent_id, observation, vehicle_conflicts=0):
        """
        Calculate local agent reward based on its observation state.
        Observation features: [veh_count, ped_count, avg_ttc, avg_speed, conflict_count]
        """
        veh_count = observation[0]
        ped_count = observation[1]
        avg_ttc = observation[2]
        avg_speed = observation[3]
        ped_conflict_count = observation[4] # Active conflicts in this step
        
        # 1. Efficiency Reward (Higher average speed is better)
        # Only reward if there are actually vehicles on the approach
        efficiency_reward = 0.0
        if veh_count > 0:
            efficiency_reward = self.w_speed * avg_speed
            
        # 2. Pedestrian Conflict Penalty
        pedestrian_conflict_penalty = -self.w_ped_conflict * ped_conflict_count
        
        # 3. Vehicle-Vehicle Conflict Penalty
        vehicle_conflict_penalty = -self.w_veh_conflict * vehicle_conflicts
        
        # 4. Low TTC Penalty (exponential penalty for near-collisions)
        ttc_penalty = 0.0
        if avg_ttc < self.ttc_safe_limit:
            # Penalize proportional to proximity (1/TTC)
            ttc_penalty = -self.w_ttc_penalty * (1.0 / max(avg_ttc, 0.1))
            
        local_reward = efficiency_reward + pedestrian_conflict_penalty + vehicle_conflict_penalty + ttc_penalty
        return local_reward

    def calculate_cooperative_reward(self, rewards_dict):
        """
        Formulate a cooperative team reward for MAPPO.
        In cooperative MARL, agents often receive the average team reward 
        to ensure they learn to coordinate and prevent greedy behaviors.
        """
        if not rewards_dict:
            return 0.0
        return sum(rewards_dict.values()) / len(rewards_dict)
