"""
Deep Q-Network (DQN) Agent with Attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Any, Optional
from .base import BaseAgent
from ..models.networks import QNetwork
from ..utils.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """DQN Agent with Flash/Linear Attention support"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # DQN specific parameters
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.tau = config.get("tau", 0.005)
        self.target_update_freq = config.get("target_update_freq", 1000)
        
        # Create Q networks
        self.q_network = QNetwork(config).to(self.device)
        self.target_q_network = QNetwork(config).to(self.device)
        
        # Copy parameters to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            size=config.get("buffer_size", 100000),
            state_dim=config["state_dim"],
            action_dim=1,  # DQN uses discrete actions
            device=self.device
        )
        
        # Training parameters
        self.update_after = config.get("update_after", 1000)
        self.update_every = config.get("update_every", 4)
        
        # Track training metrics
        self.q_losses = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            action = random.randint(0, self.config["action_dim"] - 1)
        else:
            # Greedy action
            state_tensor = self.preprocess_state(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        return np.array(action)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        # Convert action to array for buffer compatibility
        action_array = np.array([action])
        self.replay_buffer.store(state, action_array, reward, next_state, done)
    
    def update(self, batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """Update DQN agent"""
        if len(self.replay_buffer) < self.update_after:
            return {}
        
        if self.step_count % self.update_every != 0:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Update Q network
        q_loss = self._update_q_network(batch)
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            "q_loss": q_loss,
            "epsilon": self.epsilon
        }
    
    def _update_q_network(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update Q network"""
        states = batch['states']
        actions = batch['actions'].long().squeeze(-1)  # Convert to long and remove last dim
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())
        
        # Compute loss
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        self.q_losses.append(q_loss.item())
        return q_loss.item()
    
    def _update_target_network(self):
        """Update target network"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str) -> None:
        """Save agent state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
        }
        
        torch.save(checkpoint, path)
        self.save_config(path.replace('.pt', ''))
    
    def load(self, path: str) -> None:
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.episode_rewards = checkpoint['episode_rewards']
    
    def set_training_mode(self, training: bool = True):
        """Set training mode"""
        self.q_network.train(training)
    
    def to(self, device: torch.device):
        """Move agent to device"""
        super().to(device)
        self.q_network = self.q_network.to(device)
        self.target_q_network = self.target_q_network.to(device)
        return self
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        stats = super().get_stats()
        
        if self.q_losses:
            stats.update({
                "mean_q_loss": np.mean(self.q_losses[-100:]),
                "epsilon": self.epsilon,
            })
        
        return stats 