"""
Soft Actor-Critic (SAC) Agent with Attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base import BaseAgent
from ..models.networks import PolicyNetwork, QNetwork
from ..utils.replay_buffer import ReplayBuffer


class SACAgent(BaseAgent):
    """SAC Agent with Flash/Linear Attention support"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # SAC specific parameters
        self.tau = config.get("tau", 0.005)
        self.alpha = config.get("alpha", 0.2)
        self.automatic_entropy_tuning = config.get("automatic_entropy_tuning", True)
        self.target_update_interval = config.get("target_update_interval", 1)
        
        # Create networks
        self.policy = PolicyNetwork(config).to(self.device)
        
        # Q networks with double Q-learning
        q_config = config.copy()
        q_config["double_q"] = True
        self.critic = QNetwork(q_config).to(self.device)
        self.critic_target = QNetwork(q_config).to(self.device)
        
        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.get("policy_lr", 3e-4)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.get("critic_lr", 3e-4)
        )
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -config["action_dim"]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.get("alpha_lr", 3e-4))
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            size=config.get("buffer_size", 1000000),
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            device=self.device
        )
        
        # Training parameters
        self.update_after = config.get("update_after", 1000)
        self.update_every = config.get("update_every", 50)
        self.updates_per_step = config.get("updates_per_step", 1)
        
        # Track training metrics
        self.policy_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            action_dist = self.policy.get_action_distribution(state_tensor)
            
            if training:
                action = action_dist.sample()
            else:
                # Use mean action for evaluation
                action = action_dist.mean
        
        return self.postprocess_action(action)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.store(state, action, reward, next_state, done)
    
    def update(self, batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """Update SAC agent"""
        if len(self.replay_buffer) < self.update_after:
            return {}
        
        if self.step_count % self.update_every != 0:
            return {}
        
        metrics = {}
        
        for _ in range(self.updates_per_step):
            # Sample batch
            batch = self.replay_buffer.sample(self.batch_size)
            
            # Update critic
            critic_loss = self._update_critic(batch)
            metrics["critic_loss"] = critic_loss
            
            # Update policy
            policy_loss = self._update_policy(batch)
            metrics["policy_loss"] = policy_loss
            
            # Update alpha
            if self.automatic_entropy_tuning:
                alpha_loss = self._update_alpha(batch)
                metrics["alpha_loss"] = alpha_loss
                metrics["alpha"] = self.alpha
            
            # Update target networks
            if self.step_count % self.target_update_interval == 0:
                self._soft_update_target()
        
        return metrics
    
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update critic networks"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        with torch.no_grad():
            # Sample next actions from policy
            next_action_dist = self.policy.get_action_distribution(next_states)
            next_actions = next_action_dist.sample()
            next_log_probs = next_action_dist.log_prob(next_actions)
            
            # Compute target Q values
            next_q1, next_q2 = self.critic_target.forward_double(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones.float()) * next_q
        
        # Current Q values
        current_q1, current_q2 = self.critic.forward_double(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        return critic_loss.item()
    
    def _update_policy(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update policy network"""
        states = batch['states']
        
        # Sample actions from current policy
        action_dist = self.policy.get_action_distribution(states)
        actions = action_dist.rsample()  # Reparameterization trick
        log_probs = action_dist.log_prob(actions)
        
        # Q values for sampled actions
        q1, q2 = self.critic.forward_double(states, actions)
        q = torch.min(q1, q2)
        
        # Policy loss
        policy_loss = (self.alpha * log_probs - q).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.policy_losses.append(policy_loss.item())
        return policy_loss.item()
    
    def _update_alpha(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update temperature parameter alpha"""
        states = batch['states']
        
        with torch.no_grad():
            action_dist = self.policy.get_action_distribution(states)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
        
        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        self.alpha_losses.append(alpha_loss.item())
        return alpha_loss.item()
    
    def _soft_update_target(self):
        """Soft update target networks"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str) -> None:
        """Save agent state"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
        }
        
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        self.save_config(path.replace('.pt', ''))
    
    def load(self, path: str) -> None:
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.episode_rewards = checkpoint['episode_rewards']
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp()
    
    def set_training_mode(self, training: bool = True):
        """Set training mode"""
        self.policy.train(training)
        self.critic.train(training)
    
    def to(self, device: torch.device):
        """Move agent to device"""
        super().to(device)
        self.policy = self.policy.to(device)
        self.critic = self.critic.to(device)
        self.critic_target = self.critic_target.to(device)
        if self.automatic_entropy_tuning:
            self.log_alpha = self.log_alpha.to(device)
        return self
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        stats = super().get_stats()
        
        if self.policy_losses:
            stats.update({
                "mean_policy_loss": np.mean(self.policy_losses[-100:]),
                "mean_critic_loss": np.mean(self.critic_losses[-100:]),
            })
            
        if self.alpha_losses:
            stats["mean_alpha_loss"] = np.mean(self.alpha_losses[-100:])
        
        return stats 