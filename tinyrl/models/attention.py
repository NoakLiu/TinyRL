"""
Attention mechanisms for TinyRL: Flash-Attention and Linear-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from .base import BaseModel

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not available, using standard attention")


class FlashAttentionLayer(nn.Module):
    """Flash Attention layer for efficient attention computation"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        if FLASH_ATTN_AVAILABLE:
            # Use Flash Attention
            out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        else:
            # Fallback to standard attention
            out = self._standard_attention(q, k, v, mask)
            
        # Reshape and project output
        out = out.view(batch_size, seq_len, d_model)
        return self.out_proj(out)
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention as fallback"""
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        return torch.matmul(attn_weights, v)


class LinearAttentionLayer(nn.Module):
    """Linear Attention layer for O(n) complexity"""
    
    def __init__(self, d_model: int, n_heads: int, feature_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feature mapping for linear attention
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, feature_dim),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply feature mapping
        q_features = self.feature_map(q)  # [B, L, H, F]
        k_features = self.feature_map(k)  # [B, L, H, F]
        
        # Linear attention computation
        # Compute K^T V first (more efficient)
        kv = torch.einsum('blhf,blhd->bhfd', k_features, v)  # [B, H, F, D]
        
        # Then compute Q (K^T V)
        out = torch.einsum('blhf,bhfd->blhd', q_features, kv)  # [B, L, H, D]
        
        # Normalization
        k_sum = k_features.sum(dim=1, keepdim=True)  # [B, 1, H, F]
        normalizer = torch.einsum('blhf,b1hf->blh1', q_features, k_sum)  # [B, L, H, 1]
        normalizer = normalizer.clamp(min=1e-6)
        out = out / normalizer
        
        # Reshape and project output
        out = out.view(batch_size, seq_len, d_model)
        return self.out_proj(out)


class HybridAttentionLayer(nn.Module):
    """Hybrid attention that combines Flash and Linear attention"""
    
    def __init__(self, d_model: int, n_heads: int, use_flash: bool = True, 
                 use_linear: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
        self.use_linear = use_linear
        
        if self.use_flash:
            self.flash_attn = FlashAttentionLayer(d_model, n_heads, dropout)
        if self.use_linear:
            self.linear_attn = LinearAttentionLayer(d_model, n_heads)
            
        # Gating mechanism to combine outputs
        if self.use_flash and self.use_linear:
            self.gate = nn.Linear(d_model, 1)
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_flash and self.use_linear:
            # Use both and combine with gating
            flash_out = self.flash_attn(x, mask)
            linear_out = self.linear_attn(x)
            
            # Compute gate
            gate = torch.sigmoid(self.gate(x))
            return gate * flash_out + (1 - gate) * linear_out
            
        elif self.use_flash:
            return self.flash_attn(x, mask)
        elif self.use_linear:
            return self.linear_attn(x)
        else:
            raise ValueError("At least one attention mechanism must be enabled")


class FlashAttentionModel(BaseModel):
    """Complete model using Flash Attention"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.dropout = config.get("dropout", 0.1)
        
        # Input embedding
        self.input_proj = nn.Linear(config["input_dim"], self.d_model)
        
        # Attention layers
        self.layers = nn.ModuleList([
            FlashAttentionLayer(self.d_model, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, config["output_dim"])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Apply attention layers with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x, mask)
            x = norm(x + residual)
            
        # Output projection
        return self.output_proj(x)


class LinearAttentionModel(BaseModel):
    """Complete model using Linear Attention"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.feature_dim = config.get("feature_dim", 64)
        
        # Input embedding
        self.input_proj = nn.Linear(config["input_dim"], self.d_model)
        
        # Attention layers
        self.layers = nn.ModuleList([
            LinearAttentionLayer(self.d_model, self.n_heads, self.feature_dim)
            for _ in range(self.n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, config["output_dim"])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Apply attention layers with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x)
            x = norm(x + residual)
            
        # Output projection
        return self.output_proj(x)


class HybridAttentionModel(BaseModel):
    """Complete model using Hybrid Attention"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.dropout = config.get("dropout", 0.1)
        self.use_flash = config.get("use_flash", True)
        self.use_linear = config.get("use_linear", True)
        
        # Input embedding
        self.input_proj = nn.Linear(config["input_dim"], self.d_model)
        
        # Attention layers
        self.layers = nn.ModuleList([
            HybridAttentionLayer(self.d_model, self.n_heads, 
                               self.use_flash, self.use_linear, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, config["output_dim"])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Apply attention layers with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x, mask)
            x = norm(x + residual)
            
        # Output projection
        return self.output_proj(x) 