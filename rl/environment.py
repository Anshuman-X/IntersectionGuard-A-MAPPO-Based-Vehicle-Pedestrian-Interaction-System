"""
rl/environment.py
=================
Gymnasium-compatible multi-agent environment for MAPPO training at an
unsignalized T-intersection under Indian mixed traffic conditions.
Optimized with cached state queries and calibrated real-world values.
"""

import os
import sys
import numpy as np
import math

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

        # Observations (Task 2): 9-feature local state vector
        # [avg_speed, avg_pos, avg_direction, density, avg_waiting_time, step_pet, avg_ttc, gap_acceptance, crossing_status]
        self.observation_spaces = {
            agent: spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([25, 100, 360, 5, 1000, 5, 20, 10, 1], dtype=np.float32),
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
        self.pet_tracker       = LivePETTracker()

        # TraCI status flag
        self.traci_started = False

        # Episode-level logging
        self.episode_rewards: list[float] = []
        self.episode_pet_events: int = 0

    def start_sumo(self):
        """Start the SUMO simulation process via TraCI."""
        if not self.traci_started:
            sumo_binary = "sumo-gui" if self.use_gui else "sumo"
            sumo_cmd = [sumo_binary, "-c", self.sumocfg_path]
            traci.start(sumo_cmd, label="intersection_guard")
            self.traci_started = True

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

        # Reset trackers
        self.pet_tracker.reset()

        # Get initial observations with pre-populated cache
        vehicles   = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        
        self.state_extractor.update_cache(vehicles, pedestrians)

        obs = {}
        for agent in self.agents:
            obs[agent] = np.array(
                self.state_extractor.get_agent_observation(agent, vehicles, pedestrians, step_pet=5.0),
                dtype=np.float32
            )

        self.state_extractor.clear_cache()

        info = {agent: {} for agent in self.agents}
        return obs, info

    def step(self, actions_dict: dict):
        """
        Execute actions for all agents, advance the simulation, and return
        new observations, rewards, terminated, truncated, and info dicts.
        """
        self.current_step += 1

        # Gather active entities before step execution
        vehicles    = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()

        # Update cache for state extraction and action checks
        self.state_extractor.update_cache(vehicles, pedestrians)

        # Track vehicle speeds for emergency braking detection (Task 3)
        prev_speeds = {}
        for v in vehicles:
            if v in self.state_extractor.cache["veh_speed"]:
                prev_speeds[v] = self.state_extractor.cache["veh_speed"][v]

        # 1. Execute actions for each approach controller
        for agent in self.agents:
            action     = actions_dict.get(agent, 0)
            agent_vehs = self.state_extractor.get_agent_vehicles(agent, vehicles)
            agent_peds = self.state_extractor.get_agent_pedestrians(agent, pedestrians)
            self.action_executor.execute_action(agent, action, agent_vehs, agent_peds)

        # 2. Advance the SUMO simulation step
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # Gather entities in the new state
        next_vehicles    = traci.vehicle.getIDList()
        next_pedestrians = traci.person.getIDList()

        # Populate cache for the next state extraction
        self.state_extractor.update_cache(next_vehicles, next_pedestrians)

        # 3. Update PET tracker with this step's occupancy data
        self.pet_tracker.step(sim_time, next_vehicles, next_pedestrians)

        # 4. Detect safety metrics (Task 3)
        collisions = len(traci.simulation.getCollisions())
        
        emergency_brakes = {a: 0 for a in self.agents}
        veh_conflicts = {a: 0 for a in self.agents}
        ped_conflicts = {a: 0 for a in self.agents}

        for agent in self.agents:
            agent_vehs = self.state_extractor.get_agent_vehicles(agent, next_vehicles)
            
            # Detect emergency braking
            for v in agent_vehs:
                if v in prev_speeds and v in self.state_extractor.cache["veh_speed"]:
                    decel = prev_speeds[v] - self.state_extractor.cache["veh_speed"][v]
                    # deceleration > 4.5 m/s2 is emergency braking
                    if decel > 4.5:
                        emergency_brakes[agent] += 1
            
            # Detect vehicle-vehicle conflicts (TTC <= 3.0s)
            for i in range(len(agent_vehs)):
                v1 = agent_vehs[i]
                if v1 not in self.state_extractor.cache["veh_pos"] or v1 not in self.state_extractor.cache["veh_speed"]:
                    continue
                pos1 = self.state_extractor.cache["veh_pos"][v1]
                speed1 = self.state_extractor.cache["veh_speed"][v1]
                for j in range(i + 1, len(agent_vehs)):
                    v2 = agent_vehs[j]
                    if v2 not in self.state_extractor.cache["veh_pos"] or v2 not in self.state_extractor.cache["veh_speed"]:
                        continue
                    pos2 = self.state_extractor.cache["veh_pos"][v2]
                    dist = math.dist(pos1, pos2)
                    rel_speed = abs(speed1 - self.state_extractor.cache["veh_speed"][v2])
                    if dist <= 15.0 and rel_speed > 0.1:
                        ttc = dist / rel_speed
                        if ttc <= 3.0:
                            veh_conflicts[agent] += 1

            # Detect pedestrian-vehicle conflicts
            agent_peds_now = self.state_extractor.get_agent_pedestrians(agent, next_pedestrians)
            for v in agent_vehs:
                if v not in self.state_extractor.cache["veh_pos"] or v not in self.state_extractor.cache["veh_speed"]:
                    continue
                v_pos = self.state_extractor.cache["veh_pos"][v]
                v_speed = self.state_extractor.cache["veh_speed"][v]
                for p in agent_peds_now:
                    if p not in self.state_extractor.cache["ped_pos"]:
                        continue
                    p_pos = self.state_extractor.cache["ped_pos"][p]
                    dist = math.dist(v_pos, p_pos)
                    if dist <= 20.0 and v_speed > 0.1:
                        ttc_vp = dist / v_speed
                        if ttc_vp <= 3.0:
                            ped_conflicts[agent] += 1

        # 5. Extract observations and compute rewards
        obs          = {}
        local_rewards = {}

        for agent in self.agents:
            step_pet = self.pet_tracker.get_step_pet(agent)
            
            obs_vector = self.state_extractor.get_agent_observation(
                agent, next_vehicles, next_pedestrians, step_pet=step_pet
            )
            obs[agent] = np.array(obs_vector, dtype=np.float32)

            agent_vehs = self.state_extractor.get_agent_vehicles(agent, next_vehicles)
            agent_peds = self.state_extractor.get_agent_pedestrians(agent, next_pedestrians)

            # Calculate vehicle waiting time and unnecessary stopping (Task 3)
            veh_waiting_time = sum(self.state_extractor.cache["veh_waiting"].get(v, 0.0) for v in agent_vehs)
            stopped_unnecessarily = 0
            if obs_vector[8] == 0.0:  # crossing_status is 0.0
                stopped_unnecessarily = sum(1 for v in agent_vehs if self.state_extractor.cache["veh_speed"].get(v, 0.0) < 0.1)

            # Calculate local reward
            local_rewards[agent] = self.reward_calculator.calculate_reward(
                agent_id=agent,
                observation=obs_vector,
                agent_vehicles=agent_vehs,
                agent_pedestrians=agent_peds,
                collisions=collisions,
                emergency_brakes=emergency_brakes[agent],
                vehicle_conflicts=veh_conflicts[agent],
                veh_waiting_time=veh_waiting_time,
                stopped_unnecessarily=stopped_unnecessarily,
            )

        # Clear state extractor cache at step end
        self.state_extractor.clear_cache()

        # 6. Formulate cooperative reward for MAPPO
        team_reward = self.reward_calculator.calculate_cooperative_reward(local_rewards)
        rewards = {agent: team_reward for agent in self.agents}

        self.episode_rewards.append(team_reward)

        # 7. Check termination and truncation
        terminated_flag = (traci.simulation.getMinExpectedNumber() <= 0)
        truncated_flag  = (self.current_step >= self.max_steps)

        terminated = {agent: terminated_flag for agent in self.agents}
        truncated  = {agent: truncated_flag  for agent in self.agents}

        # 8. Build per-agent info dicts for diagnostic logging
        infos = {
            agent: {
                "step_conflicts": veh_conflicts[agent] + ped_conflicts[agent],
                "ped_conflicts":  ped_conflicts[agent],
                "veh_conflicts":  veh_conflicts[agent],
                "avg_speed":      obs[agent][0],
                "avg_ttc":        obs[agent][6],
                "step_pet":       obs[agent][5],
                "collisions":     collisions,
            }
            for agent in self.agents
        }

        return obs, rewards, terminated, truncated, infos

    def get_global_obs(self, obs_dict: dict) -> np.ndarray:
        """
        Concatenate all agent observations into a single global state vector
        for the CENTRALISED CRITIC in MAPPO.
        Shape: (9 × 3,) = (27,)
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
        pass


if __name__ == "__main__":
    env = IntersectionGuardEnv(use_gui=False, max_steps=5)
    print("Resetting RL environment...")
    obs, info = env.reset()
    for agent, o in obs.items():
        print(f"  * {agent} : {o}")

    random_actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    next_obs, rewards, terminated, truncated, infos = env.step(random_actions)
    print("Next step observations:")
    for agent, o in next_obs.items():
        print(f"  * {agent} : {o}")
    print(f"Rewards: {rewards}")
    env.close()
