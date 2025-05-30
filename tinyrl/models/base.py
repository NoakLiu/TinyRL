"""
Base model classes for TinyRL
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseModel(nn.Module, ABC):
    """Base class for all neural network models in TinyRL"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the model"""
        pass
    
    def save(self, path: str) -> None:
        """Save model state dict"""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str) -> None:
        """Load model state dict"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
        
    def freeze_parameters(self) -> None:
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True 