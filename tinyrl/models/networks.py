"""
Neural network architectures for RL agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .base import BaseModel
from .attention import FlashAttentionModel, LinearAttentionModel, HybridAttentionModel


class PolicyNetwork(BaseModel):
    """Policy network for actor-critic algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.attention_type = config.get("attention_type", "hybrid")
        self.use_attention = config.get("use_attention", True)
        
        if self.use_attention:
            # Use attention-based backbone
            attention_config = {
                "input_dim": self.state_dim,
                "output_dim": self.hidden_dim,
                "d_model": config.get("d_model", 256),
                "n_heads": config.get("n_heads", 8),
                "n_layers": config.get("n_layers", 4),
                "dropout": config.get("dropout", 0.1),
                "device": config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            }
            
            if self.attention_type == "flash":
                self.backbone = FlashAttentionModel(attention_config)
            elif self.attention_type == "linear":
                self.backbone = LinearAttentionModel(attention_config)
            elif self.attention_type == "hybrid":
                self.backbone = HybridAttentionModel(attention_config)
            else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
        else:
            # Use standard MLP
            self.backbone = nn.Sequential(
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # Log std for continuous actions
        if config.get("continuous_actions", False):
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
    
    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle different input shapes
        if len(state.shape) == 2:
            # Add sequence dimension for attention
            state = state.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        else:
            squeeze_output = False
            
        if self.use_attention:
            features = self.backbone(state, mask)
        else:
            features = self.backbone(state)
            
        if squeeze_output:
            features = features.squeeze(1)  # [B, D]
            
        logits = self.policy_head(features)
        return logits
    
    def get_action_distribution(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Get action distribution for sampling"""
        logits = self.forward(state, mask)
        
        if hasattr(self, 'log_std'):
            # Continuous actions - return Normal distribution
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(logits, std)
        else:
            # Discrete actions - return Categorical distribution
            return torch.distributions.Categorical(logits=logits)


class ValueNetwork(BaseModel):
    """Value network for critic"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.attention_type = config.get("attention_type", "hybrid")
        self.use_attention = config.get("use_attention", True)
        
        if self.use_attention:
            # Use attention-based backbone
            attention_config = {
                "input_dim": self.state_dim,
                "output_dim": self.hidden_dim,
                "d_model": config.get("d_model", 256),
                "n_heads": config.get("n_heads", 8),
                "n_layers": config.get("n_layers", 4),
                "dropout": config.get("dropout", 0.1),
                "device": config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            }
            
            if self.attention_type == "flash":
                self.backbone = FlashAttentionModel(attention_config)
            elif self.attention_type == "linear":
                self.backbone = LinearAttentionModel(attention_config)
            elif self.attention_type == "hybrid":
                self.backbone = HybridAttentionModel(attention_config)
            else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
        else:
            # Use standard MLP
            self.backbone = nn.Sequential(
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle different input shapes
        if len(state.shape) == 2:
            # Add sequence dimension for attention
            state = state.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        else:
            squeeze_output = False
            
        if self.use_attention:
            features = self.backbone(state, mask)
        else:
            features = self.backbone(state)
            
        if squeeze_output:
            features = features.squeeze(1)  # [B, D]
            
        value = self.value_head(features)
        return value.squeeze(-1)  # [B] or [B, L]


class QNetwork(BaseModel):
    """Q-network for value-based methods"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.attention_type = config.get("attention_type", "hybrid")
        self.use_attention = config.get("use_attention", True)
        self.double_q = config.get("double_q", False)
        
        # Input dimension includes state and action for continuous control
        if config.get("continuous_actions", False):
            input_dim = self.state_dim + self.action_dim
        else:
            input_dim = self.state_dim
            
        if self.use_attention:
            # Use attention-based backbone
            attention_config = {
                "input_dim": input_dim,
                "output_dim": self.hidden_dim,
                "d_model": config.get("d_model", 256),
                "n_heads": config.get("n_heads", 8),
                "n_layers": config.get("n_layers", 4),
                "dropout": config.get("dropout", 0.1),
                "device": config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            }
            
            if self.attention_type == "flash":
                self.backbone = FlashAttentionModel(attention_config)
                if self.double_q:
                    self.backbone2 = FlashAttentionModel(attention_config)
            elif self.attention_type == "linear":
                self.backbone = LinearAttentionModel(attention_config)
                if self.double_q:
                    self.backbone2 = LinearAttentionModel(attention_config)
            elif self.attention_type == "hybrid":
                self.backbone = HybridAttentionModel(attention_config)
                if self.double_q:
                    self.backbone2 = HybridAttentionModel(attention_config)
            else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
        else:
            # Use standard MLP
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            
            if self.double_q:
                self.backbone2 = nn.Sequential(
                    nn.Linear(input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
        
        # Q-value heads
        if config.get("continuous_actions", False):
            # For continuous actions, output single Q-value
            self.q_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )
            if self.double_q:
                self.q_head2 = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1)
                )
        else:
            # For discrete actions, output Q-value for each action
            self.q_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim)
            )
            if self.double_q:
                self.q_head2 = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.action_dim)
                )
        
        self.continuous_actions = config.get("continuous_actions", False)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.continuous_actions and action is not None:
            # Concatenate state and action for continuous control
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
            
        # Handle different input shapes
        if len(x.shape) == 2:
            # Add sequence dimension for attention
            x = x.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        else:
            squeeze_output = False
            
        if self.use_attention:
            features = self.backbone(x, mask)
        else:
            features = self.backbone(x)
            
        if squeeze_output:
            features = features.squeeze(1)  # [B, D]
            
        q_values = self.q_head(features)
        
        if not self.continuous_actions:
            return q_values  # [B, A] for discrete actions
        else:
            return q_values.squeeze(-1)  # [B] for continuous actions
    
    def forward_double(self, state: torch.Tensor, action: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for double Q-learning"""
        if not self.double_q:
            raise ValueError("Double Q-learning not enabled")
            
        if self.continuous_actions and action is not None:
            # Concatenate state and action for continuous control
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
            
        # Handle different input shapes
        if len(x.shape) == 2:
            # Add sequence dimension for attention
            x = x.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        else:
            squeeze_output = False
            
        if self.use_attention:
            features1 = self.backbone(x, mask)
            features2 = self.backbone2(x, mask)
        else:
            features1 = self.backbone(x)
            features2 = self.backbone2(x)
            
        if squeeze_output:
            features1 = features1.squeeze(1)  # [B, D]
            features2 = features2.squeeze(1)  # [B, D]
            
        q_values1 = self.q_head(features1)
        q_values2 = self.q_head2(features2)
        
        if not self.continuous_actions:
            return q_values1, q_values2  # [B, A] for discrete actions
        else:
            return q_values1.squeeze(-1), q_values2.squeeze(-1)  # [B] for continuous actions


class ActorCriticNetwork(BaseModel):
    """Combined actor-critic network"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Create separate policy and value networks
        self.policy_net = PolicyNetwork(config)
        self.value_net = ValueNetwork(config)
        
    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both policy logits and value"""
        policy_logits = self.policy_net(state, mask)
        value = self.value_net(state, mask)
        return policy_logits, value
    
    def get_action_and_value(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Get action distribution and value estimate"""
        action_dist = self.policy_net.get_action_distribution(state, mask)
        value = self.value_net(state, mask)
        return action_dist, value 