import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from rl.networks import Actor, CentralizedCritic

class MAPPO:
    """
    MAPPO (Multi-Agent Proximal Policy Optimization) implementation.
    Features:
    - Shared Actor policy network (decentralized execution)
    - Shared Centralized Critic value network (centralized training)
    - PPO clipping and GAE
    - Entropy regularisation (entropy bonus)
    - Optimization step, gradient clipping, checkpoint loading/saving
    """
    def __init__(
        self,
        obs_dim: int = 5,
        action_dim: int = 4,
        state_dim: int = 15,
        n_agents: int = 3,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize Shared Actor and Centralized Critic
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = CentralizedCritic(state_dim).to(device)
        
        # Initialize Optimizers (separate for actor and critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

    @torch.no_grad()
    def get_actions(self, obs_dict: dict, agent_ids: list) -> tuple[dict[str, int], dict[str, float]]:
        """
        Retrieves discrete actions and log-probabilities for all agents using the shared policy.
        Uses batched execution for maximum speed/parallelism.
        """
        actions = {}
        log_probs = {}
        
        # Batch observations: shape (n_agents, obs_dim)
        obs_list = [obs_dict[agent_id] for agent_id in agent_ids]
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        
        # Evaluate actions through batched actor forward pass
        action_tensor, log_prob_tensor = self.actor.get_action(obs_tensor)
        
        for i, agent_id in enumerate(agent_ids):
            actions[agent_id] = action_tensor[i].item()
            log_probs[agent_id] = log_prob_tensor[i].item()
            
        return actions, log_probs

    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """
        Queries the centralized critic value V(s) for the concatenated global state.
        """
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        value_tensor = self.critic(state_tensor)
        return value_tensor.squeeze(0).item()

    def update(self, memory) -> dict[str, float]:
        """
        Updates policy and value functions using memory experience rollout.
        """
        actor_losses = []
        critic_losses = []
        entropies = []
        
        for _ in range(self.ppo_epochs):
            for batch in memory.generate_batches(self.batch_size):
                b_states, b_returns, b_obs, b_actions, b_log_probs, b_advs = batch
                
                # 1. Update Centralized Critic (Value Loss)
                values_pred = self.critic(b_states).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values_pred, b_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # 2. Update Shared Actor (Policy Loss with clipping and entropy bonus)
                new_log_probs, entropy = self.actor.evaluate_actions(b_obs, b_actions)
                
                # Ratio r_t
                ratio = torch.exp(new_log_probs - b_log_probs)
                
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                
                entropy_loss = -entropy.mean()
                actor_loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
                
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies))
        }

    def save(self, filepath: str):
        """
        Saves Actor and Critic checkpoint weights and optimizer states.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """
        Loads Actor and Critic checkpoint weights and optimizer states.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
