"""
rl/reward.py
============
Reward calculator for cooperative MAPPO agents at an unsignalized T-intersection.

Reward Philosophy
-----------------
The reward function is the "objective" that MAPPO learns to maximize. Every term
must have a clear physical meaning tied to traffic safety:

    reward = efficiency_reward
           + pet_reward
           - pedestrian_conflict_penalty
           - vehicle_conflict_penalty
           - ttc_penalty

Term-by-Term Explanation
-------------------------

1. efficiency_reward  = w_speed × avg_speed
   - Prevents the agent from learning the trivial policy of "stop all vehicles forever."
   - Higher average speed (within safe bounds) means less congestion and better throughput.
   - Weight w_speed = 0.2 is intentionally small so safety terms dominate.

2. pet_reward  = w_pet × avg_pet
   - PET (Post-Encroachment Time) is the temporal gap between a vehicle leaving a
     conflict zone and a pedestrian entering the same zone.
   - Larger PET → safer temporal separation → positive reward signal.
   - When no PET event occurs, avg_pet defaults to PET_THRESHOLD (a safe value),
     so the agent is rewarded for maintaining separation even when no conflict is active.
   - Weight w_pet = 1.5 (moderate positive).

3. pedestrian_conflict_penalty  = -w_ped_conflict × conflict_count
   - Penalises every active vehicle-pedestrian TTC conflict detected this step.
   - Heaviest weight (10.0) because pedestrian safety is the primary objective.
   - Directly incentivises Actions 2 and 3 (Stop / Prioritize Pedestrians).

4. vehicle_conflict_penalty  = -w_veh_conflict × vehicle_conflicts
   - Penalises vehicle-vehicle near-misses at the intersection.
   - Lighter weight (2.0) than pedestrian penalty — vehicles can negotiate at
     lower risk than vehicle-pedestrian encounters.

5. ttc_penalty  = -w_ttc × (1 / avg_ttc)  when avg_ttc < ttc_safe_limit
   - This is an inverse-TTC penalty: as TTC → 0, the penalty grows very steeply.
   - Only applied when TTC falls below the safety threshold (3 seconds),
     so normal free-flow speeds are not penalised.
   - Formula: 1/TTC grows from 0.33 (at TTC=3s) to 10 (at TTC=0.1s).

Cooperative Reward
------------------
calculate_cooperative_reward() averages all agent local rewards into one team reward.
All agents receive this same value, which is the MAPPO paradigm:
  - MAPPO trains a CENTRALISED CRITIC on the joint reward.
  - Each DECENTRALISED ACTOR acts on its own local observation.
  - Shared reward encourages cooperative behaviour rather than greedy local optima.

Paper Reference
---------------
Reward shaping following: Yu et al. (2022) "The Surprising Effectiveness of MAPPO"
TTC-based safety penalty following: Songchitruksa & Tarko (2006) PVCA framework.
"""


class RewardCalculator:
    """
    Computes per-step rewards for cooperative MAPPO agents.
    Balances safety (minimising conflicts, maximising PET/TTC) and
    efficiency (maximising throughput).
    """

    def __init__(
        self,
        w_speed: float        = 0.2,
        w_pet: float          = 1.5,
        w_ped_conflict: float = 20.0,
        w_veh_conflict: float = 5.0,
        w_ttc_penalty: float  = 3.0,
        ttc_safe_limit: float = 3.0,
        pet_threshold: float  = 5.0,
    ):
        self.w_speed         = w_speed
        self.w_pet           = w_pet
        self.w_ped_conflict  = w_ped_conflict
        self.w_veh_conflict  = w_veh_conflict
        self.w_ttc_penalty   = w_ttc_penalty
        self.ttc_safe_limit  = ttc_safe_limit
        self.pet_threshold   = pet_threshold

    def calculate_reward(
        self,
        agent_id: str,
        observation: list,
        vehicle_conflicts: int = 0,
        avg_pet: float | None  = None,
    ) -> float:
        """
        Calculate local reward for one agent based on its 5-feature observation.

        Observation layout (must match state.py):
            [0] veh_count      – number of vehicles on this approach
            [1] ped_count      – pedestrians in the crossing zone
            [2] avg_ttc        – average TTC across active conflicts (s)
            [3] avg_speed      – average vehicle speed (m/s)
            [4] conflict_count – active TTC-based conflict count this step

        Parameters
        ----------
        agent_id         : Identifier (for future per-agent weight tuning).
        observation      : 5-element list/array from StateExtractor.
        vehicle_conflicts: Global vehicle-vehicle conflicts at junction this step.
        avg_pet          : Average PET for this agent's zone this step (seconds).
                           If None, defaults to pet_threshold (no conflict = safe).
        """
        veh_count        = observation[0]
        ped_count        = observation[1]   # noqa: F841 – kept for future penalty term
        avg_ttc          = observation[2]
        avg_speed        = observation[3]
        ped_conflict_cnt = observation[4]

        # Use safe default PET when no event was detected this step
        if avg_pet is None:
            avg_pet = self.pet_threshold

        # ── 1. Efficiency reward ─────────────────────────────────────────────
        efficiency_reward = 0.0
        if veh_count > 0:
            efficiency_reward = self.w_speed * avg_speed

        # ── 2. PET reward ────────────────────────────────────────────────────
        pet_reward = self.w_pet * avg_pet

        # ── 3. Pedestrian conflict penalty ───────────────────────────────────
        pedestrian_conflict_penalty = -self.w_ped_conflict * ped_conflict_cnt

        # ── 4. Vehicle-vehicle conflict penalty ──────────────────────────────
        vehicle_conflict_penalty = -self.w_veh_conflict * vehicle_conflicts

        # ── 5. Low-TTC inverse penalty ───────────────────────────────────────
        ttc_penalty = 0.0
        if avg_ttc < self.ttc_safe_limit:
            # max(..., 0.1) prevents division-by-zero when TTC is near zero
            ttc_penalty = -self.w_ttc_penalty * (1.0 / max(avg_ttc, 0.1))

        local_reward = (
            efficiency_reward
            + pet_reward
            + pedestrian_conflict_penalty
            + vehicle_conflict_penalty
            + ttc_penalty
        )
        return local_reward

    def calculate_cooperative_reward(self, rewards_dict: dict) -> float:
        """
        Formulate a shared cooperative reward for MAPPO.

        All agents receive the average team reward so that no single agent is
        incentivised to optimise its local approach at the expense of another.
        This is the key design principle of MAPPO cooperative training.

        Returns
        -------
        float : Mean of all agent local rewards.
        """
        if not rewards_dict:
            return 0.0
        return sum(rewards_dict.values()) / len(rewards_dict)
