"""
Base agent class for TinyRL
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import os
import json


class BaseAgent(ABC):
    """Base class for all RL agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        # Training parameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 256)
        
        # Logging
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        
        # Model saving
        self.save_dir = config.get("save_dir", "./checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
        
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action given state"""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update agent parameters"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent state"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent state"""
        pass
    
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state for neural network input"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        
        # Add batch dimension if needed
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        return state
    
    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        """Postprocess action for environment"""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Remove batch dimension if single action
        if action.shape[0] == 1:
            action = action.squeeze(0)
            
        return action
    
    def log_episode(self, episode_reward: float, episode_length: int, info: Dict[str, Any] = None):
        """Log episode statistics"""
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        
        # Keep only recent episodes for memory efficiency
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        if not self.episode_rewards:
            return {}
            
        return {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
            "std_episode_reward": np.std(self.episode_rewards[-100:]),
            "max_episode_reward": np.max(self.episode_rewards[-100:]),
            "min_episode_reward": np.min(self.episode_rewards[-100:]),
        }
    
    def save_config(self, path: str) -> None:
        """Save agent configuration"""
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_config = {}
            for k, v in self.config.items():
                if isinstance(v, (str, int, float, bool, list, dict)):
                    serializable_config[k] = v
                else:
                    serializable_config[k] = str(v)
            json.dump(serializable_config, f, indent=2)
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load agent configuration"""
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def set_training_mode(self, training: bool = True):
        """Set training mode for all networks"""
        # Override in subclasses to set network training modes
        pass
    
    def to(self, device: torch.device):
        """Move agent to device"""
        self.device = device
        # Override in subclasses to move networks to device
        return self 