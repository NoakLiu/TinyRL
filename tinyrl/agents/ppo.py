"""
Proximal Policy Optimization (PPO) Agent with Attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from .base import BaseAgent
from ..models.networks import ActorCriticNetwork
from ..utils.replay_buffer import PPOBuffer


class PPOAgent(BaseAgent):
    """PPO Agent with Flash/Linear Attention support"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # PPO specific parameters
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.mini_batch_size = config.get("mini_batch_size", 64)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate,
            eps=config.get("adam_eps", 1e-5)
        )
        
        # Learning rate scheduler
        if config.get("use_lr_scheduler", False):
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.get("total_timesteps", 1000000) // config.get("rollout_length", 2048)
            )
        else:
            self.lr_scheduler = None
        
        # Buffer for storing rollouts
        self.buffer = PPOBuffer(
            size=config.get("rollout_length", 2048),
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma
        )
        
        # Track training metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            action_dist, value = self.actor_critic.get_action_and_value(state_tensor)
            
            if training:
                action = action_dist.sample()
            else:
                # Use mean action for evaluation
                if hasattr(action_dist, 'mean'):
                    action = action_dist.mean
                else:
                    action = action_dist.probs.argmax(dim=-1)
            
            log_prob = action_dist.log_prob(action)
            
        # Store for buffer if training
        if training and hasattr(self, '_last_state'):
            self.buffer.store(
                state=self._last_state,
                action=self._last_action,
                reward=self._last_reward,
                value=self._last_value,
                log_prob=self._last_log_prob,
                done=self._last_done
            )
        
        # Store current step info
        self._last_state = state
        self._last_action = action
        self._last_value = value
        self._last_log_prob = log_prob
        self._last_reward = 0.0  # Will be updated in step
        self._last_done = False
        
        return self.postprocess_action(action)
    
    def step(self, reward: float, done: bool):
        """Update with environment step result"""
        self._last_reward = reward
        self._last_done = done
        self.step_count += 1
        
        if done:
            # Store final transition
            if hasattr(self, '_last_state'):
                self.buffer.store(
                    state=self._last_state,
                    action=self._last_action,
                    reward=self._last_reward,
                    value=self._last_value,
                    log_prob=self._last_log_prob,
                    done=self._last_done
                )
    
    def update(self, batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """Update PPO agent"""
        if not self.buffer.is_full():
            return {}
        
        # Compute advantages and returns
        self.buffer.compute_advantages_and_returns()
        
        # Get all data from buffer
        data = self.buffer.get_all()
        
        # Normalize advantages
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(data['states']))
            
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch
                batch_states = data['states'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = data['returns'][batch_indices]
                
                # Forward pass
                action_dist, values = self.actor_critic.get_action_and_value(batch_states)
                
                # Compute policy loss
                new_log_probs = action_dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Compute entropy loss
                entropy = action_dist.entropy().mean()
                entropy_loss = -self.entropy_coef * entropy
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Clear buffer
        self.buffer.clear()
        
        # Store metrics
        self.policy_losses.extend(policy_losses)
        self.value_losses.extend(value_losses)
        self.entropy_losses.extend(entropy_losses)
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def save(self, path: str) -> None:
        """Save agent state"""
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)
        self.save_config(path.replace('.pt', ''))
    
    def load(self, path: str) -> None:
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.episode_rewards = checkpoint['episode_rewards']
        
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    
    def set_training_mode(self, training: bool = True):
        """Set training mode"""
        self.actor_critic.train(training)
    
    def to(self, device: torch.device):
        """Move agent to device"""
        super().to(device)
        self.actor_critic = self.actor_critic.to(device)
        self.buffer.device = device
        return self
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        stats = super().get_stats()
        
        if self.policy_losses:
            stats.update({
                "mean_policy_loss": np.mean(self.policy_losses[-100:]),
                "mean_value_loss": np.mean(self.value_losses[-100:]),
                "mean_entropy_loss": np.mean(self.entropy_losses[-100:]),
            })
        
        return stats 