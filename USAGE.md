# TinyRL 使用指南

## 快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/your-username/TinyRL.git
cd TinyRL

# 安装依赖
pip install -r requirements.txt

# 安装框架
pip install -e .
```

### 2. 运行测试

```bash
python test_framework.py
```

### 3. 训练第一个模型

```bash
python examples/train_ppo_cartpole.py
```

## 核心概念

### 注意力机制

TinyRL支持三种注意力机制：

1. **Flash Attention**: 内存高效的标准注意力
2. **Linear Attention**: O(n)复杂度的线性注意力  
3. **Hybrid Attention**: 动态组合Flash和Linear注意力

### 支持的算法

- **PPO**: 适合连续和离散动作空间
- **SAC**: 适合连续动作空间
- **DQN**: 适合离散动作空间

## 详细使用

### 配置系统

TinyRL使用字典配置系统：

```python
config = {
    # 环境配置
    "env_name": "CartPole-v1",
    "state_dim": 4,
    "action_dim": 2,
    "continuous_actions": False,
    
    # 网络配置
    "hidden_dim": 256,
    "attention_type": "hybrid",  # "flash", "linear", "hybrid", None
    "use_attention": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 0.1,
    
    # 算法配置
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    
    # 训练配置
    "device": "cuda",  # "cuda", "cpu", "auto"
}
```

### 使用PPO

```python
from tinyrl.agents.ppo import PPOAgent
import gymnasium as gym

# 创建配置
config = {
    "state_dim": 4,
    "action_dim": 2,
    "continuous_actions": False,
    "attention_type": "hybrid",
    "use_attention": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "rollout_length": 2048,
}

# 创建智能体
agent = PPOAgent(config)

# 训练循环
env = gym.make("CartPole-v1")
state, _ = env.reset()

for step in range(100000):
    # 选择动作
    action = agent.select_action(state, training=True)
    
    # 环境步进
    next_state, reward, done, truncated, info = env.step(action)
    
    # 更新智能体
    agent.step(reward, done or truncated)
    
    if done or truncated:
        # 如果缓冲区满了就更新
        if agent.buffer.is_full():
            update_info = agent.update()
            print(f"Step {step}: {update_info}")
        
        state, _ = env.reset()
    else:
        state = next_state
```

### 使用SAC

```python
from tinyrl.agents.sac import SACAgent

# SAC配置（连续动作）
config = {
    "state_dim": 3,
    "action_dim": 1,
    "continuous_actions": True,
    "attention_type": "linear",
    "use_attention": True,
    "tau": 0.005,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "buffer_size": 1000000,
    "update_after": 1000,
    "update_every": 50,
}

agent = SACAgent(config)

# 训练循环
env = gym.make("Pendulum-v1")
state, _ = env.reset()

for step in range(100000):
    action = agent.select_action(state, training=True)
    next_state, reward, done, truncated, info = env.step(action)
    
    # 存储转换
    agent.store_transition(state, action, reward, next_state, done or truncated)
    
    # 更新智能体
    if step > 1000:  # 开始更新
        update_info = agent.update()
        if update_info:
            print(f"Step {step}: {update_info}")
    
    if done or truncated:
        state, _ = env.reset()
    else:
        state = next_state
```

### 使用DQN

```python
from tinyrl.agents.dqn import DQNAgent

# DQN配置（离散动作）
config = {
    "state_dim": 4,
    "action_dim": 2,
    "continuous_actions": False,
    "attention_type": "flash",
    "use_attention": True,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 100000,
    "update_after": 1000,
    "update_every": 4,
    "target_update_freq": 1000,
}

agent = DQNAgent(config)

# 训练循环
env = gym.make("CartPole-v1")
state, _ = env.reset()

for step in range(100000):
    action = agent.select_action(state, training=True)
    next_state, reward, done, truncated, info = env.step(action)
    
    # 存储转换
    agent.store_transition(state, action, reward, next_state, done or truncated)
    
    # 更新智能体
    update_info = agent.update()
    if update_info:
        print(f"Step {step}: {update_info}")
    
    if done or truncated:
        state, _ = env.reset()
    else:
        state = next_state
```

## 注意力机制详解

### Flash Attention

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

**优点**:
- 内存高效
- 计算速度快
- 支持长序列

**适用场景**:
- GPU内存充足
- 需要处理中等长度序列
- 追求训练速度

### Linear Attention

```python
config = {
    "attention_type": "linear",
    "use_attention": True,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "feature_dim": 64,  # Linear attention特有参数
}
```

**优点**:
- O(n)时间复杂度
- 内存使用极低
- 适合超长序列

**适用场景**:
- 内存受限环境
- 需要处理很长序列
- CPU训练

### Hybrid Attention

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

**优点**:
- 自适应选择最佳注意力
- 平衡性能和效率
- 动态门控机制

**适用场景**:
- 不确定最佳注意力类型
- 需要最佳性能
- 序列长度变化较大

## 环境配置

### 自动配置

```python
from tinyrl.envs.env_utils import create_env_config

# 自动推断环境配置
config = create_env_config("CartPole-v1")
print(config)
# {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'continuous_actions': False, 'max_episode_steps': 500}
```

### 手动配置

```python
config = {
    "env_name": "YourCustomEnv-v1",
    "state_dim": 10,
    "action_dim": 3,
    "continuous_actions": True,
    "max_episode_steps": 1000,
}
```

## 日志和监控

### 使用内置Logger

```python
from tinyrl.utils.logger import Logger

logger = Logger("./logs")

# 记录指标
logger.log({"reward": 100, "loss": 0.1}, step=1000)

# 获取统计信息
stats = logger.get_stats("reward")
print(stats)

# 保存配置
logger.save_config(config)

# 关闭logger
logger.close()
```

### 使用Wandb

```python
import wandb

# 初始化wandb
wandb.init(project="tinyrl-experiment", config=config)

# 训练循环中记录
wandb.log({
    "reward": episode_reward,
    "loss": loss_value,
    "step": step
})
```

## 模型保存和加载

### 保存模型

```python
# 保存检查点
agent.save("./checkpoints/model_step_10000.pt")

# 保存最终模型
agent.save("./checkpoints/final_model.pt")
```

### 加载模型

```python
# 加载模型
agent.load("./checkpoints/model_step_10000.pt")

# 设置为评估模式
agent.set_training_mode(False)

# 评估
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

## 性能优化

### GPU使用

```python
config["device"] = "cuda"  # 使用GPU
config["device"] = "cpu"   # 使用CPU
config["device"] = "auto"  # 自动选择
```

### 批处理大小

```python
# 根据GPU内存调整
config["batch_size"] = 256      # 标准
config["batch_size"] = 512      # 大内存GPU
config["batch_size"] = 128      # 小内存GPU
```

### 注意力优化

```python
# 减少注意力层数
config["n_layers"] = 2

# 减少注意力头数
config["n_heads"] = 4

# 减少模型维度
config["d_model"] = 128
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 使用linear attention
   - 减少网络层数

2. **训练不稳定**
   - 降低学习率
   - 增加梯度裁剪
   - 检查奖励缩放

3. **收敛慢**
   - 增加网络容量
   - 调整超参数
   - 使用预训练模型

### 调试技巧

```python
# 打印网络参数数量
print(f"Total parameters: {agent.actor_critic.get_num_params():,}")

# 检查梯度
for name, param in agent.actor_critic.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# 监控损失
stats = agent.get_stats()
print(stats)
```

## 扩展开发

### 添加新算法

参考现有算法实现，继承BaseAgent类：

```python
from tinyrl.agents.base import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # 初始化你的算法
        
    def select_action(self, state, training=True):
        # 实现动作选择
        pass
        
    def update(self, batch=None):
        # 实现参数更新
        pass
        
    def save(self, path):
        # 实现模型保存
        pass
        
    def load(self, path):
        # 实现模型加载
        pass
```

### 添加新注意力机制

参考现有注意力实现：

```python
from tinyrl.models.base import BaseModel

class MyAttentionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # 实现你的注意力机制
        
    def forward(self, x, mask=None):
        # 实现前向传播
        return output
``` 