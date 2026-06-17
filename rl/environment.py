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

class IntersectionGuardEnv(gym.Env):
    """
    A Multi-Agent Gymnasium-compatible environment for cooperative intersection management.
    Controls vehicle flows at an unsignalized T-intersection to prevent vehicle-pedestrian conflicts.
    Designed for MAPPO (Multi-Agent Proximal Policy Optimization) training.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "IntersectionGuard-v0"}

    def __init__(self, sumocfg_path="simulation/t_intersection.sumocfg", use_gui=False, max_steps=1000):
        super().__init__()
        
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0
        
        # 1. Define cooperative agents
        self.agents = ["west_controller", "east_controller", "north_controller"]
        
        # 2. Define action and observation spaces for each agent
        # Actions: 0 = Do Nothing, 1 = Slow, 2 = Stop, 3 = Prioritize Pedestrians (Yield)
        self.action_spaces = {
            agent: spaces.Discrete(4) for agent in self.agents
        }
        
        # Observations: [vehicle_count, pedestrian_count, average_ttc, average_speed, conflict_count]
        self.observation_spaces = {
            agent: spaces.Box(
                low=[0.0, 0.0, 0.0, 0.0, 0.0],
                high=[100.0, 100.0, 3.0, 20.0, 50.0],
                shape=(5,),
                dtype=np.float32
            ) for agent in self.agents
        }
        
        # For single-agent gymnasium API compliance, we expose joint spaces
        self.action_space = spaces.Dict(self.action_spaces)
        self.observation_space = spaces.Dict(self.observation_spaces)
        
        # 3. Instantiate helper modules
        self.state_extractor = StateExtractor()
        self.action_executor = ActionExecutor()
        self.reward_calculator = RewardCalculator()
        
        # TraCI status flag
        self.traci_started = False

    def start_sumo(self):
        """Start the SUMO simulation process via TraCI."""
        if not self.traci_started:
            sumo_binary = "sumo-gui" if self.use_gui else "sumo"
            sumo_cmd = [sumo_binary, "-c", self.sumocfg_path]
            # Label connection to support multiple client bindings in RL libraries
            traci.start(sumo_cmd, label="intersection_guard")
            self.traci_started = True

    def reset(self, seed=None, options=None):
        """Reset the environment state and start a new simulation episode."""
        self.current_step = 0
        
        # Restart or initialize TraCI
        if self.traci_started:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self.traci_started = False
            
        self.start_sumo()
        
        # Get initial observations
        vehicles = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        
        obs = {}
        for agent in self.agents:
            obs[agent] = np.array(
                self.state_extractor.get_agent_observation(agent, vehicles, pedestrians),
                dtype=np.float32
            )
            
        info = {agent: {} for agent in self.agents}
        return obs, info

    def step(self, actions_dict):
        """
        Execute actions for all agents, step the simulation, and return new state, 
        reward, terminated, truncated, and info dictionaries.
        """
        self.current_step += 1
        
        # Gather active entities before step execution
        vehicles = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        
        # 1. Execute actions for each approach controller
        for agent in self.agents:
            action = actions_dict.get(agent, 0) # Default to 'do nothing' if missing
            agent_vehs = self.state_extractor.get_agent_vehicles(agent, vehicles)
            agent_peds = self.state_extractor.get_agent_pedestrians(agent, pedestrians)
            self.action_executor.execute_action(agent, action, agent_vehs, agent_peds)
            
        # 2. Advance the SUMO simulation step
        traci.simulationStep()
        
        # Gather entities in the new state
        next_vehicles = traci.vehicle.getIDList()
        next_pedestrians = traci.person.getIDList()
        
        # 3. Extract observations and compute rewards
        obs = {}
        local_rewards = {}
        
        # Standard SUMO-level vehicle-to-vehicle conflicts inside the intersection
        global_veh_conflicts = 0
        
        for agent in self.agents:
            obs_vector = self.state_extractor.get_agent_observation(agent, next_vehicles, next_pedestrians)
            obs[agent] = np.array(obs_vector, dtype=np.float32)
            
            # Calculate local reward for this agent
            local_rewards[agent] = self.reward_calculator.calculate_reward(
                agent_id=agent,
                observation=obs_vector,
                vehicle_conflicts=global_veh_conflicts
            )
            
        # 4. Formulate cooperative reward for MAPPO
        team_reward = self.reward_calculator.calculate_cooperative_reward(local_rewards)
        rewards = {agent: team_reward for agent in self.agents}
        
        # 5. Check termination and truncation
        terminated_flag = (traci.simulation.getMinExpectedNumber() <= 0)
        truncated_flag = (self.current_step >= self.max_steps)
        
        terminated = {agent: terminated_flag for agent in self.agents}
        truncated = {agent: truncated_flag for agent in self.agents}
        
        # Expose metadata in info dict
        infos = {
            agent: {
                "step_conflicts": obs_vector[4],
                "avg_speed": obs_vector[3]
            } for agent, obs_vector in zip(self.agents, [self.state_extractor.get_agent_observation(a, next_vehicles, next_pedestrians) for a in self.agents])
        }
        
        return obs, rewards, terminated, truncated, infos

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
    if USING_MOCK_GYM:
        print("Note: 'gymnasium' or 'gym' is not installed. Running in Compatibility Mock Mode.")
    else:
        print("Running in Native Gym/Gymnasium Mode.")
        
    # Smoke-test of the RL environment setup
    env = IntersectionGuardEnv(use_gui=False, max_steps=10)
    print("Resetting RL environment...")
    obs, info = env.reset()
    print("Initial Observations:")
    for agent, o in obs.items():
        print(f"  * {agent} : {o}")
        
    print("\nExecuting a single random control step...")
    # Sample random actions for each agent
    random_actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    print(f"Random Actions: {random_actions}")
    
    next_obs, rewards, terminated, truncated, infos = env.step(random_actions)
    print("Next Step Observations:")
    for agent, o in next_obs.items():
        print(f"  * {agent} : {o}")
    print(f"Rewards: {rewards}")
    
    env.close()
    print("Environment smoke-test completed successfully!")
