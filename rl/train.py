import os
import sys
import csv
import time
import argparse
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[Warning] TensorBoard is not installed. TensorBoard logs will not be saved.")

# Ensure SUMO_HOME is set
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("Please set the 'SUMO_HOME' environment variable to run the simulation.")

from rl.environment import IntersectionGuardEnv
from rl.memory import Memory
from rl.mappo import MAPPO

def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO on IntersectionGuard-v0")
    parser.add_argument("--episodes", type=int, default=500, help="Total training episodes (default: 500)")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    parser.add_argument("--n_steps", type=int, default=256, help="Rollout steps before PPO update (default: 256)")
    parser.add_argument("--use_gui", action="store_true", help="Launch SUMO-GUI instead of CLI sumo")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate (default: 3e-4)")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate (default: 1e-3)")
    return parser.parse_args()

def init_csv_log(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "total_reward",
            "avg_reward_per_step",
            "avg_ttc",
            "avg_pet",
            "total_conflicts",
            "episode_length",
            "critic_loss",
            "actor_loss_west",
            "actor_loss_east",
            "actor_loss_north",
            "entropy_west",
            "entropy_east",
            "entropy_north",
            "duration_s",
        ])

def log_episode_csv(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path configuration
    model_dir = "models"
    tb_dir = "tensorboard"
    csv_log_path = "results/training_log.csv"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    # Initialize environment
    env = IntersectionGuardEnv(
        use_gui=args.use_gui,
        max_steps=args.steps
    )
    
    agent_ids = env.agents
    n_agents = len(agent_ids)
    obs_dim = 9
    state_dim = obs_dim * n_agents  # 27
    action_dim = 4
    
    # Initialize MAPPO and Memory
    mappo = MAPPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        n_agents=n_agents,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        device=device
    )
    
    memory = Memory(
        n_steps=args.n_steps,
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        device=device
    )
    
    # Initialize Loggers
    writer = SummaryWriter(log_dir=tb_dir) if HAS_TENSORBOARD else None
    init_csv_log(csv_log_path)
    
    print(f"\n============================================================")
    print(f"  IntersectionGuard MAPPO Training Pipeline")
    print(f"============================================================")
    print(f"  Device           : {device}")
    print(f"  Episodes         : {args.episodes}")
    print(f"  Max Steps/Ep     : {args.steps}")
    print(f"  Rollout Buffer   : {args.n_steps}")
    print(f"  Actor LR         : {args.lr_actor}")
    print(f"  Critic LR        : {args.lr_critic}")
    print(f"  Tensorboard Logs : {tb_dir}/")
    print(f"  Model Output     : {model_dir}/")
    print(f"============================================================\n")
    
    best_reward = -float("inf")
    
    for episode in range(args.episodes):
        ep_start_time = time.time()
        
        obs, _ = env.reset()
        global_obs = env.get_global_obs(obs)
        
        ep_total_reward = 0.0
        ep_steps = 0
        ep_ttc_vals = []
        ep_conflicts = 0
        
        # Track training losses for logging
        last_metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        
        done = False
        while not done:
            # 1. Select actions and evaluate value
            actions_dict, log_probs_dict = mappo.get_actions(obs, agent_ids)
            value = mappo.get_value(global_obs)
            
            # 2. Step environment
            next_obs, rewards, terminated, truncated, infos = env.step(actions_dict)
            next_global_obs = env.get_global_obs(next_obs)
            
            # Cooperative team reward is shared (same for all agents)
            team_reward = rewards[agent_ids[0]]
            ep_total_reward += team_reward
            
            # Accumulate safety metrics
            for agent_id in agent_ids:
                info = infos[agent_id]
                ep_ttc_vals.append(info.get("avg_ttc", 3.0))
                ep_conflicts += info.get("step_conflicts", 0)
                
            ep_steps += 1
            
            # Check done flags
            is_terminated = any(terminated[a] for a in agent_ids)
            is_truncated = any(truncated[a] for a in agent_ids)
            done = is_terminated or is_truncated
            
            # 3. Format dictionaries to numpy arrays for storage
            obs_array = np.array([obs[a] for a in agent_ids])
            action_array = np.array([actions_dict[a] for a in agent_ids])
            log_prob_array = np.array([log_probs_dict[a] for a in agent_ids])
            next_obs_array = np.array([next_obs[a] for a in agent_ids])
            
            # 4. Store in buffer
            memory.store(
                state=global_obs,
                obs=obs_array,
                action=action_array,
                reward=team_reward,
                next_state=next_global_obs,
                next_obs=next_obs_array,
                done=done,
                log_prob=log_prob_array,
                value=value
            )
            
            # Roll forward observation pointers
            obs = next_obs
            global_obs = next_global_obs
            
            # 5. PPO update when memory buffer is full
            if memory.is_full():
                last_value = 0.0 if done else mappo.get_value(global_obs)
                memory.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)
                last_metrics = mappo.update(memory)
                memory.reset()
                
        # 6. Force partial update at end of episode if leftover data exists
        if memory.ptr > 0:
            last_value = 0.0
            memory.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)
            last_metrics = mappo.update(memory)
            memory.reset()
            
        # Post-episode metric processing
        ep_duration = time.time() - ep_start_time
        avg_step_reward = ep_total_reward / max(ep_steps, 1)
        avg_ttc = float(np.mean(ep_ttc_vals)) if ep_ttc_vals else 3.0
        avg_pet = env.pet_tracker.get_episode_avg_pet()
        pet_count = env.pet_tracker.get_episode_pet_count()
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("reward/total", ep_total_reward, episode + 1)
            writer.add_scalar("reward/avg_per_step", avg_step_reward, episode + 1)
            writer.add_scalar("safety/avg_ttc", avg_ttc, episode + 1)
            writer.add_scalar("safety/avg_pet", avg_pet, episode + 1)
            writer.add_scalar("safety/pet_events", pet_count, episode + 1)
            writer.add_scalar("safety/total_conflicts", ep_conflicts, episode + 1)
            writer.add_scalar("losses/actor_loss", last_metrics["actor_loss"], episode + 1)
            writer.add_scalar("losses/critic_loss", last_metrics["critic_loss"], episode + 1)
            writer.add_scalar("losses/entropy", last_metrics["entropy"], episode + 1)
            
        # Log to CSV
        log_episode_csv(csv_log_path, [
            episode + 1,
            round(ep_total_reward, 4),
            round(avg_step_reward, 4),
            round(avg_ttc, 4),
            round(avg_pet, 4),
            int(ep_conflicts),
            ep_steps,
            round(last_metrics["critic_loss"], 6),
            round(last_metrics["actor_loss"], 6),  # West Actor
            round(last_metrics["actor_loss"], 6),  # East Actor
            round(last_metrics["actor_loss"], 6),  # North Actor
            round(last_metrics["entropy"], 4),     # West Entropy
            round(last_metrics["entropy"], 4),     # East Entropy
            round(last_metrics["entropy"], 4),     # North Entropy
            round(ep_duration, 2)
        ])
        
        # Print episode status
        print(
            f"Ep {episode+1:>4}/{args.episodes} | "
            f"Reward: {ep_total_reward:+8.2f} | "
            f"TTC: {avg_ttc:.2f}s | "
            f"PET: {avg_pet:.2f}s | "
            f"Conflicts: {int(ep_conflicts):>4} | "
            f"Steps: {ep_steps:>4} | "
            f"ActorL: {last_metrics['actor_loss']:.4f} | "
            f"CriticL: {last_metrics['critic_loss']:.4f} | "
            f"Time: {ep_duration:.1f}s"
        )
        
        # Save checkpoints every 50 episodes
        if (episode + 1) % 50 == 0:
            ckpt_path = os.path.join(model_dir, f"model_ep_{episode+1}.pt")
            mappo.save(ckpt_path)
            print(f"  [Checkpoint] Saved periodic model checkpoint to {ckpt_path}")
            
        # Track and save best model
        if ep_total_reward > best_reward:
            best_reward = ep_total_reward
            best_model_path = os.path.join(model_dir, "best_model.pt")
            mappo.save(best_model_path)
            print(f"  [Best Model] New best reward {best_reward:.2f}! Saved model to {best_model_path}")
            
    # Cleanup env
    env.close()
    if writer is not None:
        writer.close()
        
    print(f"\n============================================================")
    print(f"  Training Completed Successfully!")
    print(f"============================================================")
    print(f"  Best Episode Reward : {best_reward:.2f}")
    print(f"  Log File            : {csv_log_path}")
    print(f"  Models Saved In     : {model_dir}/")
    print(f"============================================================\n")

if __name__ == "__main__":
    main()
