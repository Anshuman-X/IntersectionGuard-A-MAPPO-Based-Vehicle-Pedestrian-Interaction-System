"""
rl/reward.py
============
Reward calculator for cooperative MAPPO agents at an unsignalized T-intersection.
Calibrated and improved to balance safety and throughput.
"""

class RewardCalculator:
    """
    Computes per-step rewards for cooperative MAPPO agents.
    Balances safety (minimising conflicts, emergency braking, collisions, PET/TTC) and
    efficiency (minimising waiting times, unnecessary stopping, and maximising throughput).
    """

    def __init__(
        self,
        w_speed: float        = 1.0,    # Weight for throughput/speed
        w_pet_safe: float     = 0.5,    # Weight for maintaining safe PET
        w_pet_unsafe: float   = 10.0,   # Weight for penalising unsafe PET
        w_ttc_safe: float     = 0.5,    # Weight for maintaining safe TTC
        w_ttc_unsafe: float   = 5.0,    # Weight for penalising unsafe TTC
        w_ped_wait: float     = 1.0,    # Weight for pedestrian waiting time
        w_veh_wait: float     = 0.5,    # Weight for vehicle waiting time
        w_emergency_brake: float = 15.0,# Weight for emergency braking
        w_stopped_unnecessarily: float = 10.0, # Weight for unnecessary stopping
        w_veh_conflict: float = 5.0,    # Weight for vehicle-vehicle conflicts
        w_collision: float    = 250.0,  # Weight for collisions
        ttc_safe_limit: float = 3.0,
        pet_threshold: float  = 5.0,
    ):
        self.w_speed = w_speed
        self.w_pet_safe = w_pet_safe
        self.w_pet_unsafe = w_pet_unsafe
        self.w_ttc_safe = w_ttc_safe
        self.w_ttc_unsafe = w_ttc_unsafe
        self.w_ped_wait = w_ped_wait
        self.w_veh_wait = w_veh_wait
        self.w_emergency_brake = w_emergency_brake
        self.w_stopped_unnecessarily = w_stopped_unnecessarily
        self.w_veh_conflict = w_veh_conflict
        self.w_collision = w_collision
        self.ttc_safe_limit = ttc_safe_limit
        self.pet_threshold = pet_threshold

    def calculate_reward(
        self,
        agent_id: str,
        observation: list,
        agent_vehicles: list,
        agent_pedestrians: list,
        collisions: int = 0,
        emergency_brakes: int = 0,
        vehicle_conflicts: int = 0,
        veh_waiting_time: float = 0.0,
        stopped_unnecessarily: int = 0,
    ) -> float:
        """
        Calculate local reward for one agent based on its 9-feature observation and safety metrics.

        Observation layout (must match state.py):
            [0] avg_speed        - average speed of vehicles (m/s)
            [1] avg_pos          - average distance to junction (m)
            [2] avg_direction    - average heading angle
            [3] density          - density of vehicles
            [4] avg_waiting_time - average pedestrian waiting time (s)
            [5] step_pet         - step PET (s)
            [6] avg_ttc          - average vehicle-pedestrian TTC (s)
            [7] gap_acceptance   - calibrated gap acceptance (s)
            [8] crossing_status  - crossing status (1.0 or 0.0)
        """
        avg_speed        = observation[0]
        density          = observation[3]
        ped_waiting_time = observation[4]
        step_pet         = observation[5]
        avg_ttc          = observation[6]
        crossing_status  = observation[8]

        # ── 1. Throughput & Efficiency Reward ──────────────────────────────────
        # Reward active vehicle throughput (speed * density proxy)
        throughput_reward = self.w_speed * (avg_speed * density * 10.0)

        # ── 2. PET Safety Terms ────────────────────────────────────────────────
        pet_reward = 0.0
        if step_pet < self.pet_threshold:
            # Unsafe PET: apply penalty proportional to how close we are to collision
            pet_reward = -self.w_pet_unsafe * (self.pet_threshold - step_pet)
        else:
            # Safe PET: reward keeping safe temporal gap
            pet_reward = self.w_pet_safe * step_pet

        # ── 3. TTC Safety Terms ────────────────────────────────────────────────
        ttc_reward = 0.0
        if avg_ttc < self.ttc_safe_limit:
            # Unsafe TTC: steep inverse penalty as TTC -> 0
            ttc_reward = -self.w_ttc_unsafe * (1.0 / max(avg_ttc, 0.1))
        else:
            # Safe TTC: reward maintaining headway
            ttc_reward = self.w_ttc_safe * avg_ttc

        # ── 4. Waiting Time Penalties ──────────────────────────────────────────
        # Penalise pedestrian delay
        ped_delay_penalty = -self.w_ped_wait * ped_waiting_time
        # Penalise vehicle delay
        veh_delay_penalty = -self.w_veh_wait * veh_waiting_time

        # ── 5. Operational Safety Penalties ────────────────────────────────────
        # Penalise emergency deceleration (emergency braking)
        brake_penalty = -self.w_emergency_brake * emergency_brakes
        # Penalise vehicle conflicts
        conflict_penalty = -self.w_veh_conflict * vehicle_conflicts
        # Penalise collisions heavily
        collision_penalty = -self.w_collision * collisions

        # ── 6. Operational Efficiency Penalties ────────────────────────────────
        # Penalise stopping vehicles when no pedestrians are crossing
        unnecessary_stopping_penalty = -self.w_stopped_unnecessarily * stopped_unnecessarily

        # Sum terms
        local_reward = (
            throughput_reward
            + pet_reward
            + ttc_reward
            + ped_delay_penalty
            + veh_delay_penalty
            + brake_penalty
            + conflict_penalty
            + collision_penalty
            + unnecessary_stopping_penalty
        )
        return local_reward

    def calculate_cooperative_reward(self, rewards_dict: dict) -> float:
        """
        Formulate a shared cooperative team reward for MAPPO.
        All agents receive the average team reward to encourage coordination.
        """
        if not rewards_dict:
            return 0.0
        return sum(rewards_dict.values()) / len(rewards_dict)
