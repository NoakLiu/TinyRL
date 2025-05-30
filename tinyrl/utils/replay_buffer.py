"""
Replay buffers for different RL algorithms
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import random


class ReplayBuffer:
    """Standard replay buffer for off-policy algorithms"""
    
    def __init__(self, size: int, state_dim: int, action_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        
        # Initialize buffers
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)
        
    def store(self, state: np.ndarray, action: np.ndarray, reward: float, 
              next_state: np.ndarray, done: bool):
        """Store a transition"""
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.FloatTensor(next_state).to(self.device)
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions"""
        max_idx = self.size if self.full else self.ptr
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self):
        return self.size if self.full else self.ptr


class PPOBuffer:
    """Buffer for PPO algorithm with GAE computation"""
    
    def __init__(self, size: int, state_dim: int, action_dim: int, device: torch.device,
                 gae_lambda: float = 0.95, gamma: float = 0.99):
        self.size = size
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.ptr = 0
        
        # Initialize buffers
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)
        
        # Computed during finalize
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        
    def store(self, state: np.ndarray, action: torch.Tensor, reward: float,
              value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """Store a transition"""
        if self.ptr >= self.size:
            return  # Buffer is full
            
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def compute_advantages_and_returns(self, last_value: float = 0.0):
        """Compute GAE advantages and returns"""
        # Convert last_value to tensor
        last_value = torch.tensor(last_value, dtype=torch.float32, device=self.device)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[step].float()
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step].float()
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        # Compute returns
        returns = advantages + self.values[:self.ptr]
        
        self.advantages[:self.ptr] = advantages
        self.returns[:self.ptr] = returns
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all stored data"""
        return {
            'states': self.states[:self.ptr],
            'actions': self.actions[:self.ptr],
            'rewards': self.rewards[:self.ptr],
            'values': self.values[:self.ptr],
            'log_probs': self.log_probs[:self.ptr],
            'dones': self.dones[:self.ptr],
            'advantages': self.advantages[:self.ptr],
            'returns': self.returns[:self.ptr]
        }
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.ptr >= self.size
    
    def __len__(self):
        return self.ptr


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay buffer"""
    
    def __init__(self, size: int, state_dim: int, action_dim: int, device: torch.device,
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        super().__init__(size, state_dim, action_dim, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Priority tree (simplified implementation using list)
        self.priorities = np.zeros(size, dtype=np.float32)
        
    def store(self, state: np.ndarray, action: np.ndarray, reward: float,
              next_state: np.ndarray, done: bool):
        """Store transition with maximum priority"""
        super().store(state, action, reward, next_state, done)
        
        # Assign maximum priority to new experience
        self.priorities[self.ptr - 1 if self.ptr > 0 else self.size - 1] = self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, np.ndarray]:
        """Sample batch with prioritized sampling"""
        max_idx = self.size if self.full else self.ptr
        
        # Compute sampling probabilities
        priorities = self.priorities[:max_idx] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(max_idx, batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (max_idx * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        self.priorities[indices] = priorities + 1e-6  # Small epsilon to avoid zero priority
        self.max_priority = max(self.max_priority, priorities.max())


class SequenceReplayBuffer:
    """Replay buffer for sequence data (useful for attention mechanisms)"""
    
    def __init__(self, size: int, state_dim: int, action_dim: int, seq_len: int, device: torch.device):
        self.size = size
        self.seq_len = seq_len
        self.device = device
        self.ptr = 0
        self.full = False
        
        # Initialize buffers for sequences
        self.states = torch.zeros((size, seq_len, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, seq_len, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size, seq_len), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size, seq_len), dtype=torch.bool, device=device)
        self.masks = torch.zeros((size, seq_len), dtype=torch.bool, device=device)  # Valid timesteps
        
    def store_sequence(self, states: List[np.ndarray], actions: List[np.ndarray],
                      rewards: List[float], dones: List[bool]):
        """Store a complete sequence"""
        seq_length = min(len(states), self.seq_len)
        
        # Pad or truncate sequence
        for i in range(seq_length):
            self.states[self.ptr, i] = torch.FloatTensor(states[i]).to(self.device)
            self.actions[self.ptr, i] = torch.FloatTensor(actions[i]).to(self.device)
            self.rewards[self.ptr, i] = rewards[i]
            self.dones[self.ptr, i] = dones[i]
            self.masks[self.ptr, i] = True
        
        # Clear remaining positions
        for i in range(seq_length, self.seq_len):
            self.masks[self.ptr, i] = False
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of sequences"""
        max_idx = self.size if self.full else self.ptr
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'masks': self.masks[indices]
        }
    
    def __len__(self):
        return self.size if self.full else self.ptr 