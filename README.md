# TinyRL

A lightweight reinforcement learning framework integrating Flash-Attention and Linear-Attention mechanisms.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **🔥 Efficient Attention Mechanisms**: Flash-Attention and Linear-Attention integration
- **🔄 Hybrid Attention**: Dynamic combination of different attention mechanisms
- **🎯 Multiple RL Algorithms**: PPO, SAC, DQN with attention support
- **📊 Complete Training Pipeline**: Logging, model saving, evaluation, and monitoring
- **🛠️ Modular Design**: Easy to extend and customize
- **⚡ High Performance**: Optimized for both GPU and CPU training

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Guide](#usage-guide)
- [Attention Mechanisms](#attention-mechanisms)
- [Algorithms](#algorithms)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/NoakLiu/TinyRL.git
cd TinyRL

# Install dependencies
pip install -r requirements.txt

# Install TinyRL
pip install -e .
```

### Verify Installation

```bash
# Run tests to verify installation
python test_framework.py
```

## 🚀 Quick Start

### Basic Example

```python
import torch
import gymnasium as gym
from tinyrl.agents.ppo import PPOAgent

# Configuration
config = {
    "env_name": "CartPole-v1",
    "state_dim": 4,
    "action_dim": 2,
    "continuous_actions": False,
    
    # Attention configuration
    "attention_type": "hybrid",  # "flash", "linear", "hybrid"
    "use_attention": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    
    # Training parameters
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "rollout_length": 2048,
}

# Create environment and agent
env = gym.make(config["env_name"])
agent = PPOAgent(config)

# Training loop
state, _ = env.reset()
for step in range(10000):
    action = agent.select_action(state, training=True)
    next_state, reward, done, truncated, info = env.step(action)
    
    agent.step(reward, done or truncated)
    
    if done or truncated:
        if agent.buffer.is_full():
            update_info = agent.update()
            print(f"Step {step}: {update_info}")
        state, _ = env.reset()
    else:
        state = next_state
```

### Run Example Scripts

```bash
# Train PPO on CartPole with hybrid attention
python examples/train_ppo_cartpole.py

# Use configuration file
python examples/train_ppo_cartpole.py --config configs/ppo_cartpole.yaml
```

## 🧠 Core Concepts

### Attention Mechanisms

TinyRL supports three types of attention mechanisms:

| Type | Complexity | Memory | Best For |
|------|------------|--------|----------|
| **Flash Attention** | O(n²) | Low | Standard sequences, GPU training |
| **Linear Attention** | O(n) | Very Low | Long sequences, CPU training |
| **Hybrid Attention** | Adaptive | Medium | Variable-length sequences |

### Supported Algorithms

- **PPO (Proximal Policy Optimization)**: On-policy, suitable for both continuous and discrete actions
- **SAC (Soft Actor-Critic)**: Off-policy, optimized for continuous control
- **DQN (Deep Q-Network)**: Off-policy, designed for discrete action spaces

## 📖 Usage Guide

### Configuration System

TinyRL uses a dictionary-based configuration system for maximum flexibility:

```python
config = {
    # Environment
    "env_name": "CartPole-v1",
    "state_dim": 4,
    "action_dim": 2,
    "continuous_actions": False,
    
    # Network Architecture
    "hidden_dim": 256,
    "attention_type": "hybrid",  # "flash", "linear", "hybrid", None
    "use_attention": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 0.1,
    
    # Algorithm Parameters
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    
    # Training
    "device": "cuda",  # "cuda", "cpu", "auto"
    "total_timesteps": 100000,
}
```

### Environment Configuration

#### Automatic Configuration

```python
from tinyrl.envs.env_utils import create_env_config

# Automatically infer environment configuration
config = create_env_config("CartPole-v1")
print(config)
# Output: {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'continuous_actions': False}
```

#### Manual Configuration

```python
config = {
    "env_name": "YourCustomEnv-v1",
    "state_dim": 10,
    "action_dim": 3,
    "continuous_actions": True,
    "max_episode_steps": 1000,
}
```

## 🔍 Attention Mechanisms

### Flash Attention

Optimized standard attention with memory efficiency.

```python
config = {
    "attention_type": "flash",
    "use_attention": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 0.1,
}
```

**Advantages:**
- Memory efficient
- Fast computation
- Supports long sequences

**Use Cases:**
- GPU training with sufficient memory
- Medium-length sequences
- Speed-critical applications

### Linear Attention

O(n) complexity attention mechanism.

```python
config = {
    "attention_type": "linear",
    "use_attention": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "feature_dim": 64,  # Linear attention specific
}
```

**Advantages:**
- O(n) time complexity
- Very low memory usage
- Excellent for long sequences

**Use Cases:**
- Memory-constrained environments
- Very long sequences
- CPU training

### Hybrid Attention

Adaptive combination of Flash and Linear attention.

```python
config = {
    "attention_type": "hybrid",
    "use_attention": True,
    "use_flash": True,
    "use_linear": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
}
```

**Advantages:**
- Adaptive attention selection
- Balanced performance and efficiency
- Dynamic gating mechanism

**Use Cases:**
- Uncertain about optimal attention type
- Variable sequence lengths
- Maximum performance requirements

## 🤖 Algorithms

### PPO (Proximal Policy Optimization)

```python
from tinyrl.agents.ppo import PPOAgent

config = {
    "state_dim": 4,
    "action_dim": 2,
    "continuous_actions": False,
    "attention_type": "hybrid",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "rollout_length": 2048,
}

agent = PPOAgent(config)
```

### SAC (Soft Actor-Critic)

```python
from tinyrl.agents.sac import SACAgent

config = {
    "state_dim": 3,
    "action_dim": 1,
    "continuous_actions": True,
    "attention_type": "linear",
    "tau": 0.005,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "buffer_size": 1000000,
}

agent = SACAgent(config)
```

### DQN (Deep Q-Network)

```python
from tinyrl.agents.dqn import DQNAgent

config = {
    "state_dim": 4,
    "action_dim": 2,
    "continuous_actions": False,
    "attention_type": "flash",
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "buffer_size": 100000,
}

agent = DQNAgent(config)
```

## 📊 Logging and Monitoring

### Built-in Logger

```python
from tinyrl.utils.logger import Logger

logger = Logger("./logs")

# Log metrics
logger.log({"reward": 100, "loss": 0.1}, step=1000)

# Get statistics
stats = logger.get_stats("reward")
print(stats)

# Save configuration
logger.save_config(config)
logger.close()
```

### Weights & Biases Integration

```python
import wandb

# Initialize wandb
wandb.init(project="tinyrl-experiment", config=config)

# Log during training
wandb.log({
    "reward": episode_reward,
    "loss": loss_value,
    "step": step
})
```

## 💾 Model Saving and Loading

### Save Models

```python
# Save checkpoint
agent.save("./checkpoints/model_step_10000.pt")

# Save final model
agent.save("./checkpoints/final_model.pt")
```

### Load Models

```python
# Load model
agent.load("./checkpoints/model_step_10000.pt")

# Set to evaluation mode
agent.set_training_mode(False)

# Evaluate
state, _ = env.reset()
total_reward = 0

while True:
    action = agent.select_action(state, training=False)
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    
    if done or truncated:
        break

print(f"Evaluation reward: {total_reward}")
```

## ⚡ Performance Optimization

### GPU Configuration

```python
config["device"] = "cuda"  # Use GPU
config["device"] = "cpu"   # Use CPU
config["device"] = "auto"  # Auto-select
```

### Memory Optimization

```python
# Adjust batch size based on GPU memory
config["batch_size"] = 256      # Standard
config["batch_size"] = 512      # High-memory GPU
config["batch_size"] = 128      # Low-memory GPU

# Reduce attention complexity
config["n_layers"] = 2          # Fewer layers
config["n_heads"] = 4           # Fewer heads
config["d_model"] = 128         # Smaller model
```

### Performance Comparison

| Attention Type | Memory Usage | Training Speed | Long Sequence Performance |
|----------------|--------------|----------------|---------------------------|
| Flash          | Low          | Fast           | Good                      |
| Linear         | Very Low     | Very Fast      | Excellent                 |
| Hybrid         | Medium       | Medium         | Best                      |
| MLP            | Lowest       | Fastest        | Poor                      |

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size`
   - Use `linear` attention
   - Decrease network layers

2. **Training Instability**
   - Lower learning rate
   - Increase gradient clipping
   - Check reward scaling

3. **Slow Convergence**
   - Increase network capacity
   - Tune hyperparameters
   - Use pretrained models

### Debugging Tips

```python
# Print network parameters
print(f"Total parameters: {agent.actor_critic.get_num_params():,}")

# Check gradients
for name, param in agent.actor_critic.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Monitor losses
stats = agent.get_stats()
print(stats)
```

## 📁 Project Structure

```
TinyRL/
├── tinyrl/
│   ├── models/          # Neural network models
│   │   ├── attention.py # Attention mechanisms
│   │   ├── networks.py  # Network architectures
│   │   └── base.py      # Base model class
│   ├── agents/          # RL algorithms
│   │   ├── ppo.py       # PPO implementation
│   │   ├── sac.py       # SAC implementation
│   │   ├── dqn.py       # DQN implementation
│   │   └── base.py      # Base agent class
│   ├── utils/           # Utilities
│   │   ├── replay_buffer.py # Experience replay
│   │   └── logger.py    # Logging utilities
│   └── envs/            # Environment utilities
├── examples/            # Example scripts
├── configs/             # Configuration files
├── tests/               # Test files
└── docs/                # Documentation
```

## 🔬 Extending TinyRL

### Adding New Algorithms

```python
from tinyrl.agents.base import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your algorithm
        
    def select_action(self, state, training=True):
        # Implement action selection
        pass
        
    def update(self, batch=None):
        # Implement parameter updates
        pass
        
    def save(self, path):
        # Implement model saving
        pass
        
    def load(self, path):
        # Implement model loading
        pass
```

### Adding New Attention Mechanisms

```python
from tinyrl.models.base import BaseModel

class MyAttentionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Implement your attention mechanism
        
    def forward(self, x, mask=None):
        # Implement forward pass
        return output
```

## 📚 Examples

### CartPole with PPO

```bash
python examples/train_ppo_cartpole.py
```

### Pendulum with SAC

```python
from tinyrl.agents.sac import SACAgent
import gymnasium as gym

config = {
    "state_dim": 3,
    "action_dim": 1,
    "continuous_actions": True,
    "attention_type": "linear",
    "buffer_size": 1000000,
}

env = gym.make("Pendulum-v1")
agent = SACAgent(config)

# Training loop
state, _ = env.reset()
for step in range(100000):
    action = agent.select_action(state, training=True)
    next_state, reward, done, truncated, info = env.step(action)
    
    agent.store_transition(state, action, reward, next_state, done or truncated)
    
    if step > 1000:
        update_info = agent.update()
        if update_info:
            print(f"Step {step}: {update_info}")
    
    if done or truncated:
        state, _ = env.reset()
    else:
        state = next_state
```

## 📋 Requirements

- torch>=2.0.0
- numpy>=1.21.0
- gymnasium>=0.28.0
- flash-attn>=2.0.0
- transformers>=4.20.0
- wandb>=0.13.0
- tensorboard>=2.10.0
- tqdm>=4.64.0
- pyyaml>=6.0
- hydra-core>=1.2.0
- omegaconf>=2.2.0

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use TinyRL in your research, please cite:

```bibtex
@misc{tinyrl2024,
  title={TinyRL: Flash-Attention and Linear-Attention Integrated RL Framework},
  author={TinyRL Team},
  year={2024},
  url={https://github.com/NoakLiu/TinyRL}
}
```

## 🙏 Acknowledgments

- [Flash-Attention](https://github.com/Dao-AILab/flash-attention) for efficient attention implementation
- [OpenAI Gymnasium](https://gymnasium.farama.org/) for RL environments
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📞 Support

- 📧 Email: [dong.liu.dl2357@gmail.com](dong.liu.dl2357@yale.edu)
<!-- - 💬 Discord: [TinyRL Community](https://discord.gg/tinyrl) -->
- 🐛 Issues: [GitHub Issues](https://github.com/NoakLiu/TinyRL/issues)
<!-- - 📖 Documentation: [docs.tinyrl.org](https://docs.tinyrl.org) -->

---

<div align="center">
  <strong>Built with ❤️ by the TinyRL Team</strong>
</div>
