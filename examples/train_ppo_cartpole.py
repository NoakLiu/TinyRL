"""
Example training script for PPO with attention on CartPole
"""

import torch
import gymnasium as gym
import numpy as np
from tinyrl.agents.ppo import PPOAgent
from tinyrl.utils.logger import Logger
import wandb
from tqdm import tqdm


def main():
    # Configuration
    config = {
        # Environment
        "env_name": "CartPole-v1",
        "state_dim": 4,
        "action_dim": 2,
        "continuous_actions": False,
        
        # Network architecture
        "hidden_dim": 256,
        "attention_type": "hybrid",  # "flash", "linear", "hybrid", or None for MLP
        "use_attention": True,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "dropout": 0.1,
        
        # PPO parameters
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "ppo_epochs": 4,
        "mini_batch_size": 64,
        "rollout_length": 2048,
        
        # Training
        "total_timesteps": 100000,
        "eval_freq": 5000,
        "save_freq": 10000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # Logging
        "use_wandb": True,
        "project_name": "tinyrl-cartpole",
        "run_name": "ppo-hybrid-attention",
    }
    
    # Initialize wandb
    if config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            name=config["run_name"],
            config=config
        )
    
    # Create environment
    env = gym.make(config["env_name"])
    eval_env = gym.make(config["env_name"])
    
    # Create agent
    agent = PPOAgent(config)
    
    # Create logger
    logger = Logger(config.get("log_dir", "./logs"))
    
    print(f"Training PPO with {config['attention_type']} attention on {config['env_name']}")
    print(f"Device: {config['device']}")
    print(f"Total parameters: {agent.actor_critic.get_num_params():,}")
    
    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    pbar = tqdm(total=config["total_timesteps"], desc="Training")
    
    for step in range(config["total_timesteps"]):
        # Select action
        action = agent.select_action(state, training=True)
        
        # Environment step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update agent
        agent.step(reward, done)
        
        episode_reward += reward
        episode_length += 1
        
        if done:
            # Log episode
            agent.log_episode(episode_reward, episode_length, info)
            
            # Update agent if buffer is full
            if agent.buffer.is_full():
                update_info = agent.update()
                
                # Log training metrics
                if config["use_wandb"] and update_info:
                    wandb.log({
                        "train/episode_reward": episode_reward,
                        "train/episode_length": episode_length,
                        "train/policy_loss": update_info.get("policy_loss", 0),
                        "train/value_loss": update_info.get("value_loss", 0),
                        "train/entropy_loss": update_info.get("entropy_loss", 0),
                        "train/learning_rate": update_info.get("learning_rate", 0),
                        "train/step": step,
                    })
            
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
        else:
            state = next_state
        
        # Evaluation
        if step % config["eval_freq"] == 0 and step > 0:
            eval_rewards = evaluate_agent(agent, eval_env, num_episodes=10)
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            
            print(f"\nStep {step}: Eval reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            if config["use_wandb"]:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/step": step,
                })
        
        # Save model
        if step % config["save_freq"] == 0 and step > 0:
            agent.save(f"./checkpoints/ppo_step_{step}.pt")
        
        pbar.update(1)
        pbar.set_postfix({
            "Episode": episode_count,
            "Reward": f"{episode_reward:.1f}",
            "Length": episode_length
        })
    
    pbar.close()
    
    # Final evaluation
    final_rewards = evaluate_agent(agent, eval_env, num_episodes=100)
    print(f"\nFinal evaluation: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    
    # Save final model
    agent.save("./checkpoints/ppo_final.pt")
    
    env.close()
    eval_env.close()
    
    if config["use_wandb"]:
        wandb.finish()


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance"""
    agent.set_training_mode(False)
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    agent.set_training_mode(True)
    return rewards


if __name__ == "__main__":
    main() 