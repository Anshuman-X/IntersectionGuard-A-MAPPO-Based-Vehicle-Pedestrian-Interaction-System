"""
rl/pet_tracker.py
=================
Live Post-Encroachment Time (PET) tracker for use inside the MAPPO training loop.

What is PET?
------------
PET is the time gap between when a vehicle **exits** a conflict zone and when a
pedestrian **enters** the same zone (or vice-versa). A small PET means the two road
users narrowly missed sharing the same space — a near-miss even if TTC was never
triggered.

    PET = t_pedestrian_entry − t_vehicle_exit   (seconds)

Why track it live during RL?
----------------------------
The reward function needs real-time PET values so MAPPO can learn to increase the
temporal separation between vehicles and pedestrians at each crossing zone.

Design
------
This class mirrors the logic in pvca/pet.py but is designed for step-by-step use:
  - No TraCI start/stop — the environment already manages that.
  - Maintains state dicts between steps: who is in which zone, when they left.
  - Exposes get_step_pet(agent_id) to return the most recent PET for that zone.
  - Exposes reset() to clear state at episode boundaries.
"""

import traci

# These zone bounding boxes must match pvca/pet.py and rl/state.py EXACTLY.
# Junction centre is at (100.0, 0.0) in compiled SUMO network coordinates.
CONFLICT_ZONES = {
    "west_controller":  {"x_min": 92.8,  "x_max": 96.8,  "y_min": -3.5, "y_max": 3.5},
    "east_controller":  {"x_min": 103.2, "x_max": 107.2, "y_min": -3.5, "y_max": 3.5},
    "north_controller": {"x_min": 96.5,  "x_max": 103.5, "y_min":  3.2, "y_max": 7.2},
}

# Maximum PET (seconds) that still counts as a safety-relevant conflict.
# Values above this threshold mean the road users were far enough apart in time.
PET_THRESHOLD = 5.0


class LivePETTracker:
    """
    Tracks Post-Encroachment Time (PET) per agent zone during a running
    SUMO simulation. Call step() once per simulation step to update internal
    state, then read get_step_pet() for the reward calculator.

    Designed to be owned by IntersectionGuardEnv and reset on episode boundaries.
    """

    def __init__(self, pet_threshold: float = PET_THRESHOLD):
        self.pet_threshold = pet_threshold
        self.agent_ids = list(CONFLICT_ZONES.keys())

        # Persistent state across steps — reset at episode start
        # occupied_by_veh[agent_id] = set of vehicle IDs currently inside that zone
        self.occupied_by_veh: dict[str, set] = {a: set() for a in self.agent_ids}
        # occupied_by_ped[agent_id] = set of pedestrian IDs currently inside that zone
        self.occupied_by_ped: dict[str, set] = {a: set() for a in self.agent_ids}
        # last_vehicle_exit[agent_id] = (vehicle_id, exit_sim_time) or None
        self.last_vehicle_exit: dict[str, tuple | None] = {a: None for a in self.agent_ids}

        # Most recently measured PET per zone (used by reward calculator this step)
        # Default: PET_THRESHOLD (safe — no conflict measured yet)
        self.current_pet: dict[str, float] = {a: pet_threshold for a in self.agent_ids}

        # Cumulative PET events this episode (for logging)
        self.episode_pet_events: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """
        Clear all zone state at the start of a new episode.
        Must be called from IntersectionGuardEnv.reset().
        """
        self.occupied_by_veh  = {a: set() for a in self.agent_ids}
        self.occupied_by_ped  = {a: set() for a in self.agent_ids}
        self.last_vehicle_exit = {a: None  for a in self.agent_ids}
        self.current_pet = {a: self.pet_threshold for a in self.agent_ids}
        self.episode_pet_events = []

    def step(self, sim_time: float, vehicles: list[str], pedestrians: list[str]):
        """
        Process one simulation step.

        Parameters
        ----------
        sim_time    : Current SUMO simulation time in seconds.
        vehicles    : List of active vehicle IDs from traci.vehicle.getIDList().
        pedestrians : List of active pedestrian IDs from traci.person.getIDList().

        Updates self.current_pet for each agent zone.
        """
        # Build current zone occupancy for this step
        current_veh_occ = {a: set() for a in self.agent_ids}
        current_ped_occ = {a: set() for a in self.agent_ids}

        for v in vehicles:
            try:
                pos = traci.vehicle.getPosition(v)
                agent_id = self._get_zone(pos[0], pos[1])
                if agent_id:
                    current_veh_occ[agent_id].add(v)
            except traci.exceptions.TraCIException:
                continue

        for p in pedestrians:
            try:
                pos = traci.person.getPosition(p)
                agent_id = self._get_zone(pos[0], pos[1])
                if agent_id:
                    current_ped_occ[agent_id].add(p)
            except traci.exceptions.TraCIException:
                continue

        # Reset current PET to safe default each step
        # (only overwritten if a genuine PET event is detected)
        for a in self.agent_ids:
            self.current_pet[a] = self.pet_threshold

        for agent_id in self.agent_ids:
            # --- Detect vehicle exits from zone ---
            for v in list(self.occupied_by_veh[agent_id]):
                if v not in current_veh_occ[agent_id]:
                    # Vehicle just left the zone — record its exit time
                    self.last_vehicle_exit[agent_id] = (v, sim_time)

            # --- Detect pedestrian entries into zone ---
            for p in current_ped_occ[agent_id]:
                if p not in self.occupied_by_ped[agent_id]:
                    # Pedestrian just entered the zone — check for preceding vehicle
                    if self.last_vehicle_exit[agent_id] is not None:
                        v_id, exit_time = self.last_vehicle_exit[agent_id]
                        pet = sim_time - exit_time
                        if 0.0 < pet <= self.pet_threshold:
                            # Valid PET conflict detected
                            self.current_pet[agent_id] = pet
                            self.episode_pet_events.append({
                                "sim_time":   sim_time,
                                "agent":      agent_id,
                                "vehicle_id": v_id,
                                "ped_id":     p,
                                "pet":        round(pet, 2),
                            })
                            # Consume the vehicle exit record (prevents double-matching)
                            self.last_vehicle_exit[agent_id] = None

            # Update history sets for the next step
            self.occupied_by_veh[agent_id] = current_veh_occ[agent_id].copy()
            self.occupied_by_ped[agent_id] = current_ped_occ[agent_id].copy()

    def get_step_pet(self, agent_id: str) -> float:
        """
        Return the most recent PET value for the given agent zone.

        Returns self.pet_threshold (safe default) when no PET conflict occurred
        in this step, so the reward function rewards increasing PET proportionally.
        """
        return self.current_pet.get(agent_id, self.pet_threshold)

    def get_episode_avg_pet(self) -> float:
        """
        Return the average PET across all events this episode.
        Useful for end-of-episode logging.
        """
        if not self.episode_pet_events:
            return self.pet_threshold
        return round(sum(e["pet"] for e in self.episode_pet_events) / len(self.episode_pet_events), 3)

    def get_episode_pet_count(self) -> int:
        """Total number of PET conflict events recorded this episode."""
        return len(self.episode_pet_events)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_zone(self, x: float, y: float) -> str | None:
        """Return the agent_id whose crossing zone contains (x, y), or None."""
        for agent_id, bbox in CONFLICT_ZONES.items():
            if bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]:
                return agent_id
        return None
