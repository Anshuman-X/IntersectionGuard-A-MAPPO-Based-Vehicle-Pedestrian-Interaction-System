import torch
import torch.nn as nn
from torch.distributions import Categorical

def init_weights(module: nn.Module, gain: float = 1.0):
    """
    Orthogonal weight initialization for PPO training stability.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)

class Actor(nn.Module):
    """
    Actor network mapping a local 9D observation vector to action logits
    over 4 actions. Shared across all 3 agents.
    """
    def __init__(self, obs_dim: int = 9, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(lambda m: init_weights(m, gain=1.0))
        # Keep initial action distribution near uniform by reducing final layer gain
        init_weights(self.net[-1], gain=0.01)

    def forward(self, obs: torch.Tensor) -> Categorical:
        """
        Returns a Categorical distribution over the discrete actions.
        """
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action and returns (action, log_prob)
        """
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the log probability and entropy of actions under the current policy.
        Used during PPO updates.
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy

class CentralizedCritic(nn.Module):
    """
    Centralized Critic network mapping the 27D global state vector (concatenated
    observations of all 3 agents) to a scalar value estimate V(s).
    """
    def __init__(self, state_dim: int = 27, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(lambda m: init_weights(m, gain=1.0))
        init_weights(self.net[-1], gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns value estimate V(s) of shape (..., 1)
        """
        return self.net(state)
