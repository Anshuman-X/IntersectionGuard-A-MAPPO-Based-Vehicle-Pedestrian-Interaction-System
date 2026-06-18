"""
rl/environment.py
=================
Gymnasium-compatible multi-agent environment for MAPPO training at an
unsignalized T-intersection under Indian mixed traffic conditions.

Architecture
------------
Three cooperative agents — one per approach road:
  - west_controller  : Controls vehicles approaching from the west
  - east_controller  : Controls vehicles approaching from the east
  - north_controller : Controls vehicles approaching from the north (T-branch)

Each agent:
  - OBSERVES  : A 5-feature local state vector (see state.py)
  - ACTS      : One of 4 discrete actions (see actions.py)
  - RECEIVES  : A shared cooperative team reward (see reward.py)

MAPPO Compatibility
-------------------
This environment follows the Gymnasium API (reset/step/close) and is directly
compatible with the custom MAPPO trainer in mappo_trainer.py.

  - observation_space : Dict of Box(5,) per agent  — for decentralised actors
  - action_space      : Dict of Discrete(4) per agent

The CENTRALISED CRITIC concatenates all agents' observations into a 15-D global
state vector inside mappo_trainer.py — no changes needed here.

Connected Modules
-----------------
  state.py        → StateExtractor      : Reads SUMO via TraCI → observation vectors
  actions.py      → ActionExecutor      : Translates discrete actions → TraCI commands
  reward.py       → RewardCalculator    : Computes safety/efficiency reward
  pet_tracker.py  → LivePETTracker      : Tracks live PET per zone during training
"""

import os
import sys
import numpy as np


# Ensure SUMO tools path is set up
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError("Please set the 'SUMO_HOME' environment variable to run the simulation.")

import traci

# Gymnasium/Gym API Fallback Configuration for Environment Portability
try:
    import gymnasium as gym
    from gymnasium import spaces
    USING_MOCK_GYM = False
except ImportError:
    try:
        import gym
        from gym import spaces
        USING_MOCK_GYM = False
    except ImportError:
        # Create lightweight mock representations of Gym spaces so code is runnable
        # without external ML libraries installed on the system.
        class MockDiscrete:
            def __init__(self, n):
                self.n = n
            def sample(self):
                return int(np.random.randint(0, self.n))

        class MockBox:
            def __init__(self, low, high, shape, dtype):
                self.low = np.array(low, dtype=dtype)
                self.high = np.array(high, dtype=dtype)
                self.shape = shape
                self.dtype = dtype
            def sample(self):
                return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

        class MockDict:
            def __init__(self, spaces_dict):
                self.spaces = spaces_dict

        class spaces:
            Discrete = MockDiscrete
            Box = MockBox
            Dict = MockDict

        class MockEnv:
            def __init__(self):
                pass
            def reset(self, seed=None):
                pass
            def step(self, actions):
                pass

        class gym:
            Env = MockEnv

        USING_MOCK_GYM = True

from rl.state import StateExtractor
from rl.actions import ActionExecutor
from rl.reward import RewardCalculator
from rl.pet_tracker import LivePETTracker


class IntersectionGuardEnv(gym.Env):
    """
    A Multi-Agent Gymnasium-compatible environment for cooperative intersection management.
    Controls vehicle flows at an unsignalized T-intersection to prevent vehicle-pedestrian
    conflicts. Designed for MAPPO (Multi-Agent Proximal Policy Optimization) training.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "IntersectionGuard-v0"}

    def __init__(self, sumocfg_path="simulation/t_intersection.sumocfg", use_gui=False, max_steps=1000):
        super().__init__()

        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0

        # 1. Define cooperative agents (one per approach road)
        self.agents = ["west_controller", "east_controller", "north_controller"]

        # 2. Define action and observation spaces for each agent
        # Actions: 0 = Do Nothing, 1 = Slow, 2 = Stop, 3 = Prioritize Pedestrians (Yield)
        self.action_spaces = {
            agent: spaces.Discrete(4) for agent in self.agents
        }

        # Observations: [vehicle_count, pedestrian_count, avg_ttc, avg_speed, conflict_count]
        self.observation_spaces = {
            agent: spaces.Box(
                low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([100, 100, 3, 20, 50], dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.agents
        }

        # For single-agent gymnasium API compliance, expose joint spaces
        self.action_space = spaces.Dict(self.action_spaces)
        self.observation_space = spaces.Dict(self.observation_spaces)

        # 3. Instantiate helper modules
        self.state_extractor   = StateExtractor()
        self.action_executor   = ActionExecutor()
        self.reward_calculator = RewardCalculator()
        self.pet_tracker       = LivePETTracker()   # Phase A: live PET tracking

        # TraCI status flag
        self.traci_started = False

        # Episode-level logging (used by MAPPO trainer for metrics)
        self.episode_rewards: list[float] = []
        self.episode_pet_events: int = 0

    # ------------------------------------------------------------------
    # TraCI lifecycle
    # ------------------------------------------------------------------

    def start_sumo(self):
        """Start the SUMO simulation process via TraCI."""
        if not self.traci_started:
            sumo_binary = "sumo-gui" if self.use_gui else "sumo"
            sumo_cmd = [sumo_binary, "-c", self.sumocfg_path]
            # Label connection to support multiple client bindings in RL libraries
            traci.start(sumo_cmd, label="intersection_guard")
            self.traci_started = True

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment state and start a new simulation episode."""
        self.current_step = 0
        self.episode_rewards = []

        # Restart or initialise TraCI
        if self.traci_started:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self.traci_started = False

        self.start_sumo()

        # Reset PET tracker for new episode
        self.pet_tracker.reset()

        # Get initial observations
        vehicles   = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()

        obs = {}
        for agent in self.agents:
            obs[agent] = np.array(
                self.state_extractor.get_agent_observation(agent, vehicles, pedestrians),
                dtype=np.float32
            )

        info = {agent: {} for agent in self.agents}
        return obs, info

    def step(self, actions_dict: dict):
        """
        Execute actions for all agents, advance the simulation, and return
        new observations, rewards, terminated, truncated, and info dicts.

        Parameters
        ----------
        actions_dict : {agent_id: action_int} — one action per agent.

        Returns
        -------
        obs         : {agent_id: np.ndarray}  — next observations
        rewards     : {agent_id: float}        — cooperative team reward
        terminated  : {agent_id: bool}         — simulation finished naturally
        truncated   : {agent_id: bool}         — max_steps reached
        infos       : {agent_id: dict}         — diagnostic metadata
        """
        self.current_step += 1

        # Gather active entities before step execution
        vehicles    = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()

        # 1. Execute actions for each approach controller
        for agent in self.agents:
            action     = actions_dict.get(agent, 0)   # Default to 'do nothing' if missing
            agent_vehs = self.state_extractor.get_agent_vehicles(agent, vehicles)
            agent_peds = self.state_extractor.get_agent_pedestrians(agent, pedestrians)
            self.action_executor.execute_action(agent, action, agent_vehs, agent_peds)

        # 2. Advance the SUMO simulation step
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # Gather entities in the new state
        next_vehicles    = traci.vehicle.getIDList()
        next_pedestrians = traci.person.getIDList()

        # 3. Update PET tracker with this step's occupancy data (Phase A)
        self.pet_tracker.step(sim_time, next_vehicles, next_pedestrians)

        # 4. Extract observations and compute rewards
        obs          = {}
        local_rewards = {}

        # Standard SUMO-level vehicle-to-vehicle conflicts inside the intersection
        global_veh_conflicts = 0

        for agent in self.agents:
            obs_vector = self.state_extractor.get_agent_observation(
                agent, next_vehicles, next_pedestrians
            )
            obs[agent] = np.array(obs_vector, dtype=np.float32)

            # Read live PET for this agent's zone from the tracker
            step_pet = self.pet_tracker.get_step_pet(agent)

            # Calculate local reward (now includes PET term)
            local_rewards[agent] = self.reward_calculator.calculate_reward(
                agent_id=agent,
                observation=obs_vector,
                vehicle_conflicts=global_veh_conflicts,
                avg_pet=step_pet,
            )

        # 5. Formulate cooperative reward for MAPPO (average of all local rewards)
        team_reward = self.reward_calculator.calculate_cooperative_reward(local_rewards)
        rewards = {agent: team_reward for agent in self.agents}

        # Track episode reward for logging
        self.episode_rewards.append(team_reward)

        # 6. Check termination and truncation
        terminated_flag = (traci.simulation.getMinExpectedNumber() <= 0)
        truncated_flag  = (self.current_step >= self.max_steps)

        terminated = {agent: terminated_flag for agent in self.agents}
        truncated  = {agent: truncated_flag  for agent in self.agents}

        # 7. Build per-agent info dicts (diagnostic metadata for trainer logging)
        next_obs_list = [
            self.state_extractor.get_agent_observation(a, next_vehicles, next_pedestrians)
            for a in self.agents
        ]
        infos = {
            agent: {
                "step_conflicts": obs_v[4],
                "avg_speed":      obs_v[3],
                "avg_ttc":        obs_v[2],
                "step_pet":       self.pet_tracker.get_step_pet(agent),
            }
            for agent, obs_v in zip(self.agents, next_obs_list)
        }

        return obs, rewards, terminated, truncated, infos

    def get_global_obs(self, obs_dict: dict) -> np.ndarray:
        """
        Concatenate all agent observations into a single global state vector
        for the CENTRALISED CRITIC in MAPPO.

        Shape: (obs_dim × n_agents,) = (5 × 3,) = (15,)

        This is called inside the MAPPO trainer — not during normal step() execution.
        """
        return np.concatenate([obs_dict[a] for a in self.agents], axis=0)

    def close(self):
        """Close the simulation and release TraCI connection."""
        if self.traci_started:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self.traci_started = False

    def render(self):
        # Rendering is handled by SUMO-GUI when use_gui=True.
        # No additional rendering implementation is needed here.
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test: run directly with  python -m rl.environment
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if USING_MOCK_GYM:
        print("Note: 'gymnasium' or 'gym' is not installed. Running in Compatibility Mock Mode.")
    else:
        print("Running in Native Gym/Gymnasium Mode.")

    env = IntersectionGuardEnv(use_gui=False, max_steps=10)
    print("Resetting RL environment...")
    obs, info = env.reset()

    print(f"Step #{'0.00':>5}Initial Observations:")
    for agent, o in obs.items():
        print(f"  * {agent} : {o}")

    print("\nExecuting a single random control step...")
    random_actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    print(f"Random Actions: {random_actions}")

    next_obs, rewards, terminated, truncated, infos = env.step(random_actions)
    print("Next Step Observations:")
    for agent, o in next_obs.items():
        print(f"  * {agent} : {o}")
    print(f"Rewards: {rewards}")
    print(f"PET Events this step: {env.pet_tracker.get_episode_pet_count()}")

    # Also test global obs for MAPPO critic
    global_obs = env.get_global_obs(next_obs)
    print(f"Global state (for MAPPO Critic): shape={global_obs.shape}, values={global_obs}")

    env.close()
    print("Environment smoke-test completed successfully!")
