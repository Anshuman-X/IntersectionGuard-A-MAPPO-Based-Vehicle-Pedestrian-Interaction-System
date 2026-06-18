import numpy as np
import torch

class Memory:
    """
    On-policy trajectory buffer for MAPPO. Stores transitions for all cooperative agents.
    Computes GAE (Generalized Advantage Estimation) advantages and discounted returns,
    and yields mini-batches for training.
    """
    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        device: str = "cpu"
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.device = device
        
        self.reset()

    def reset(self):
        """
        Clears the buffer. Called at the start of rollout collection.
        """
        self.states = np.zeros((self.n_steps, self.state_dim), dtype=np.float32)
        self.obs = np.zeros((self.n_steps, self.n_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_agents), dtype=np.int64)
        self.rewards = np.zeros((self.n_steps,), dtype=np.float32)
        self.next_states = np.zeros((self.n_steps, self.state_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.n_steps, self.n_agents, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.n_steps,), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.n_steps,), dtype=np.float32)
        
        self.advantages = np.zeros((self.n_steps,), dtype=np.float32)
        self.returns = np.zeros((self.n_steps,), dtype=np.float32)
        
        self.ptr = 0

    def store(
        self,
        state: np.ndarray,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
        log_prob: np.ndarray,
        value: float
    ):
        """
        Stores one timestep transition in the buffer.
        """
        assert self.ptr < self.n_steps, "Rollout buffer overflow!"
        self.states[self.ptr] = state
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Computes GAE (Generalized Advantage Estimation) advantages and value targets (returns)
        backward through the collected rollout.
        """
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            # δ_t (TD error)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            # GAE accumulation
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # Discounted returns are targets for Centralized Critic
        self.returns = self.advantages + self.values

        # Standardize advantages to stabilize policy update gradients
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def generate_batches(self, batch_size: int):
        """
        Generates random mini-batches for PPO updates.
        Flattens agent dimensions for the shared Actor network.
        """
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)

        # Convert arrays to PyTorch tensors
        states_t = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(self.returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(self.advantages, dtype=torch.float32, device=self.device)

        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # Centralized Critic Batch (global state size, returns size)
            b_states = states_t[batch_idx]
            b_returns = returns_t[batch_idx]

            # Actor Batch (flatten agent dimensions: B * n_agents)
            b_obs = obs_t[batch_idx].view(-1, self.obs_dim)
            b_actions = actions_t[batch_idx].view(-1)
            b_log_probs = log_probs_t[batch_idx].view(-1)

            # Repeat advantages for each agent in the team
            b_advs = advantages_t[batch_idx].unsqueeze(1).repeat(1, self.n_agents).view(-1)

            yield b_states, b_returns, b_obs, b_actions, b_log_probs, b_advs
